
{"bool": {"must": [{"nested": {"query": {"bool": {"must": [{"range": {"schedules_nested.end_date": {"lte": "2016-10-18T02:59:59", "gte": "2016-10-11T03:00:00"}}}, {"term": {"schedules_nested.obj_status": "active"}}]}}, "path": "schedules_nested"}}]}}


class OperationStack(list):
    es_keywords = {'AND': 'must', 'OR': 'should', 'AND NOT': 'must_not', 'OR NOT': 'should_not'}

    def pop(self, index=None):
        return self.es_keywords[super(OperationStack, self).pop()]


def compile_es_query(params):
    query_string = params.pop('es_q')
    # compile params as "AND conditions" on the top level of query_string
    for key, value in params.items():
        if key.startswith('_'):
            continue
        else:
            query_string += ' AND '
            query_string += ':'.join([key, value])
    query_string = _parse_nested_items(query_string)
    query_tokens = _get_tokens(query_string)
    if len(query_tokens) > 1:
        tree = _build_tree(query_tokens)
        return {'bool': _build_es_query(tree)}
    return {'bool': {'must': [_parse_term(query_string)]}}


# split query string to tokens "(", ")", "field:value", "AND", "AND NOT", "OR", "OR NOT"
def _get_tokens(values):

    tokens = []
    brackets = ['(', ')']
    buffer = ''
    keywords = ['AND', 'OR']
    in_term = False

    for item in values:

        if item == '[':
            in_term = True

        if item == ']':
            in_term = False

        if item == ' ' and buffer and not in_term:
            if buffer == 'NOT':
                tmp = tokens.pop()
                # check for avoid issue with "field_name:NOT blabla"
                if tmp in keywords:
                    tokens.append(' '.join([tmp, buffer.strip()]))
            else:
                tokens.append(buffer.strip())
            buffer = ''
            continue

        if item in brackets:
            if buffer:
                tokens.append(buffer.strip())
            tokens.append(item)
            buffer = ''
            continue

        buffer += item

    if buffer:
        tokens.append(buffer)
    return tokens


def _build_tree(tokens):

    class Node:
        def __init__(self, prev=None, next=None):
            self.prev = prev
            self.next = next
            self.values = []

        def parse(self):
            strings = []
            for value in self.values:
                if isinstance(value, Node):
                    strings.append(value.parse())
                else:
                    strings.append(value)
            return strings

    head = Node()

    for token in tokens:
        if token == '(':
            head.next = Node(head)
            head.values.append(head.next)
            head = head.next
            continue
        if token == ')':
            head = head.prev
            continue

        head.values.append(token)
    return head.parse()


def _build_es_query(values):
    aggregation = {}
    operations_stack = OperationStack()
    values_stack = []
    keywords = ['AND', 'AND NOT', 'OR', 'OR NOT']

    for value in values:
        if value in keywords:
            operations_stack.append(value)
        else:
            values_stack.append(value)

        if len(operations_stack) == 1 and len(values_stack) == 2:
            value2 = values_stack.pop()
            value1 = values_stack.pop()

            operation = operations_stack.pop()

            if isinstance(value1, list):
                value1 = {'bool': _build_es_query(value1)}

            if isinstance(value2, list):
                value2 = {'bool': _build_es_query(value2)}

            keyword_exists = aggregation.get(operation, False)

            if keyword_exists:
                _attach_item(value2, aggregation, operation)
            else:
                if operation == 'should_not':
                    _attach_item(value1, aggregation, 'should')
                    _attach_item(value2, aggregation, operation)
                elif operation == 'must_not':
                    _attach_item(value1, aggregation, 'must')
                    _attach_item(value2, aggregation, operation)
                else:
                    for item in [value1, value2]:
                        _attach_item(item, aggregation, operation)

            values_stack.append(None)

    return aggregation


# add term to existed aggregation
def _attach_item(item, aggregation, operation):

    if item is None:
        return

    # init value or get existed
    aggregation[operation] = aggregation[operation] if len(aggregation.get(operation, [])) else []

    if _is_nested(item):
        _attach_nested(item, aggregation, operation)
    elif isinstance(item, dict):
        aggregation[operation].append(item)
    else:
        aggregation[operation].append(_parse_term(item))


# parse term, on this level can be implemented rules according to range, term, match and others
# https://www.elastic.co/guide/en/elasticsearch/reference/2.1/term-level-queries.html
def _parse_term(item):
    field, value = smart_split(item)
    if '|' in value:
        values = value.split('|')
        return {'bool': {'should': [{'term': {field: value}} for value in values]}}
    if value == '_missing_':
        return {'missing': {'field': field}}
    if value.startswith('[') and value.endswith(']'):
        return _parse_range(field, value[1:len(value) - 1])
    return {'term': {field: value}}


def _parse_range(field, value):
    from_, to = list(map(lambda string: string.strip(), value.split('TO')))
    range_ = {'range': {field: {}}}

    if from_ != '_missing_':
        range_['range'][field].update({'gte': from_})

    if to != '_missing_':
        range_['range'][field].update({'lte': to})

    return range_


# attach _nested to nested_document
def _parse_nested_items(query_string):
    parsed_query_string = ''
    in_quotes = False
    for index, key in enumerate(query_string):

        if key == '"' and not in_quotes:
            in_quotes = True
            key = ''
        elif key == '"' and in_quotes:
            in_quotes = False
            key = ''

        if key == '.' and not in_quotes:
            key = '_nested.'

        parsed_query_string = parsed_query_string + key
    return parsed_query_string


def _is_nested(item):
    if isinstance(item, str):
        field, _ = smart_split(item)
        return '_nested' in field
    return False


def smart_split(item, split_key=':'):
    split_index = -1
    for index, key in enumerate(item):
        if key == split_key:
            split_index = index
            break
    return [item[0:split_index], item[split_index + 1:]]


# rules related to nested queries
# https://www.elastic.co/guide/en/elasticsearch/guide/current/nested-query.html
def _attach_nested(value, aggregation, operation):
    field, _ = smart_split(value)
    path = field.split('.')[0]
    existed_items = aggregation[operation]
    invert_operation = {'must': 'must', 'must_not': 'must',
                        'should_not': 'should', 'should': 'should'}
    for item in existed_items:
        if 'nested' in item:
            item_path = item['nested'].get('path', False)
            if item_path == path:
                item['nested']['query']['bool'][invert_operation[operation]]\
                    .append(_parse_term(value))
                break
    else:
        existed_items.append({'nested': {
            'path': path, 'query': {'bool': {invert_operation[operation]: [_parse_term(value)]}}}})
