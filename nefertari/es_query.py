

class OperationStack(list):
    es_keywords = {'AND': 'must', 'OR': 'should', 'AND NOT': 'must_not', 'OR NOT': 'should_not'}

    def pop(self, index=None):
        return self.es_keywords[super(OperationStack, self).pop()]


def compile_es_query(query_string):
    query_string = _parse_nested_items(query_string)
    query_tokens = _get_tokens(query_string)
    if len(query_tokens) > 1:
        tree = _build_tree(query_tokens)
        return {'bool': _build_es_query(tree)}
    return {'bool': {'must': [_parse_term(query_string)]}}


def _get_tokens(values):

    tokens = []
    brackets = ['(', ')']
    buffer = ''
    keywords = ['AND', 'OR']

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
                if _is_nested(value2):
                    _attach_nested(value, aggregation, operation)
                elif isinstance(value2, dict):
                    aggregation[operation].append(value2)
                else:
                    aggregation[operation].append(_parse_term(value2))
            else:
                aggregation[operation] = []

                for item in filter(lambda x: x is not None, [value1, value2]):
                    if _is_nested(item):
                        _attach_nested(item, aggregation, operation)
                    elif isinstance(item, dict):
                        aggregation[operation].append(item)
                    else:
                        aggregation[operation].append(_parse_term(item))

            values_stack.append(None)

    return aggregation


def _parse_term(item):
    field, value = item.split(':')
    if value == '_missing_':
        return {'missing': {'field': field}}

    return {'term': {field: value}}


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
        field, _ = item.split(':')
        return '_nested' in field
    return False


def _attach_nested(value, aggregation, operation):
    field, value = value.split(':')
    path = field.split('.')[0]
    existed_items = aggregation[operation]

    for item in existed_items:
        if 'nested' in item:
            item_path = item['nested'].get('path', False)
            if item_path == path:
                item['nested']['query']['bool'][operation].append({'term': {field: value}})
                break
    else:
        existed_items.append({'nested': {'path': path, 'query': {'bool': {operation: [{'term': {field: value}}]}}}})
