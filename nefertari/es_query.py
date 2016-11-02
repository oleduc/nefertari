

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


def _get_tokens(values):
    """
    split query string to tokens "(", ")", "field:value", "AND", "AND NOT", "OR", "OR NOT"
    :param values: string
    :return: array of tokens
    """
    tokens = []
    brackets = {'(', ')'}
    buffer = ''
    keywords = {'AND', 'OR'}

    for item in values:

        if item == ' ' and buffer:
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
        def __init__(self, prev=None, next_=None):
            self.prev = prev
            self.next = next_
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
    keywords = {'AND', 'AND NOT', 'OR', 'OR NOT'}

    for value in values:
        if isinstance(value, str) and value in keywords:
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


def _attach_item(item, aggregation, operation):
    """
    attach item to already existed operation in aggregation or to new operation in aggregation
    :param item: string
    :param aggregation: dict which contains aggregated terms
    :param operation: ES operation keywords {must, must_not, should, should_not}
    :return:
    """

    if item is None:
        return

    # init value or get existed
    aggregation[operation] = aggregation.get(operation, [])

    if _is_nested(item):
        _attach_nested(item, aggregation, operation)
    elif isinstance(item, dict):
        aggregation[operation].append(item)
    else:
        aggregation[operation].append(_parse_term(item))


def _parse_term(item):
    """
    parse term, on this level can be implemented rules according to range, term, match and others
    https://www.elastic.co/guide/en/elasticsearch/reference/2.1/term-level-queries.html
    :param item: string
    :return: dict which contains {'term': {field_name: field_value}
    """

    field, value = item.split(':')
    if '|' in value:
        values = value.split('|')
        return {'bool': {'should': [{'term': {field: value}} for value in values]}}
    if value == '_missing_':
        return {'missing': {'field': field}}

    return {'term': {field: value}}


def _parse_nested_items(query_string):
    """
    attach _nested to nested_document
    :param query_string: string
    :return: string with updated name for nested document, like assignments_nested for assignments
    """
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
        field, _ = item.split(':')
        return '_nested' in field
    return False


def _attach_nested(value, aggregation, operation):
    """
    apply rules related to nested queries
    https://www.elastic.co/guide/en/elasticsearch/guide/current/nested-query.html
    :param value: string
    :param aggregation: dict which contains aggregated terms
    :param operation: ES operation keywords {must, must_not, should, should_not}
    :return: None
    """

    field, value = value.split(':')
    path = field.split('.')[0]
    existed_items = aggregation[operation]
    invert_operation = {'must': 'must', 'must_not': 'must', 'should_not': 'should', 'should': 'should'}
    for item in existed_items:
        if 'nested' in item:
            item_path = item['nested'].get('path', False)
            if item_path == path:
                item['nested']['query']['bool'][invert_operation[operation]].append({'term': {field: value}})
                break
    else:
        existed_items.append({'nested': {'path': path, 'query': {'bool': {invert_operation[operation]: [{'term': {field: value}}]}}}})
