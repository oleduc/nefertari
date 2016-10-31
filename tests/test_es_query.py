from nefertari.es_query import compile_es_query, _get_tokens, _build_tree


class TestESQueryCompilation(object):

    def test_build_parse_tokens(self):
        query_string = 'item:value OR item:value'
        assert ['item:value', 'OR', 'item:value'] == _get_tokens(query_string)
        query_string = '((item:value OR item:value) AND NOT (item:OR OR NOT item:NOT)) OR true:true'
        assert _get_tokens(query_string) == ['(', '(', 'item:value', 'OR', 'item:value', ')',
                                             'AND NOT', '(', 'item:OR', 'OR NOT', 'item:NOT',
                                             ')', ')', 'OR', 'true:true']

    def test_build_tree(self):
        query_string = '(item:value OR item:value) AND ((item:value OR item:value AND complicated:false) OR (item:value OR item:value))'
        tokens = _get_tokens(query_string)
        assert [['item:value', 'OR', 'item:value'], 'AND',
                [['item:value', 'OR', 'item:value', 'AND', 'complicated:false'], 'OR',
                 ['item:value', 'OR', 'item:value']]] == _build_tree(tokens)

    def test_nested_query(self):
        query_string = 'assignments.assignee_id:someuse'
        result = compile_es_query(query_string)
        print(result)
        assert result == {
            'bool': {'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}}]}}

    def test_nested_query_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuse.user.@b.a.b.la."'
        result = compile_es_query(query_string)
        assert result == {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse.user.@b.a.b.la.'}}]}}

    def test_nested_query_and_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuser.some.last.name" ' \
                 'AND assignments.assignor_id:"changed.user.name"'

        result = compile_es_query(query_string)
        assert result == {'bool': {'must': [{'nested': {'path': 'assignments_nested', 'query': {
            'bool': {
                'must': [{'term': {'assignments_nested.assignee_id': 'someuser.some.last.name'}},
                         {'term': {'assignments_nested.assignor_id': 'changed.user.name'}}]}}}}]}}

    def test_nested_query_and(self):
        query_string = 'assignments.assignee_id:someuse AND assignments.is_completed:true'
        result = compile_es_query(query_string)
        assert result == {'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}},
                     {'term': {'assignments_nested.is_completed': 'true'}}]}},
                                                        'path': 'assignments_nested'}}]}}

    def test_nested_query_complicated(self):
        query_string = 'assignments.assignee_id:someuse AND NOT assignments.assignor_id:someusesaqk AND assignments.is_completed:true'
        result = compile_es_query(query_string)
        assert result == {'bool': {'must_not': [{'nested': {'path': 'assignments_nested', 'query': {
            'bool': {'must_not': [{'term': {'assignments_nested.assignee_id': 'someuse'}},
                                  {'term': {'assignments_nested.assignor_id': 'someusesaqk'}}]}}}}],
                                   'must': [{'nested': {'path': 'assignments_nested', 'query': {
                                       'bool': {'must': [{'term': {
                                           'assignments_nested.is_completed': 'true'}}]}}}}]}}

    def test_nested_query_inside_query(self):
        query_string = '(assignments.assignee_id:someuser OR assignments.is_completed:false AND assignments.assignor_id:another) OR owner_id:someuser'
        result = compile_es_query(query_string)
        print(result)
        assert result == {'bool': {
            'should': [{'bool': {
                        'must': [
                            {'nested': {'query': {'bool': {'must': [{'term': {'assignments_nested.assignor_id': 'another'}}]}},'path': 'assignments_nested'}}],
                        'should': [
                            {'nested': {'query': {'bool': {
                                'should': [{'term': {'assignments_nested.assignee_id': 'someuser'}}, {'term': {'assignments_nested.is_completed': 'false'}}]}}, 'path': 'assignments_nested'}}]
                        }},
                      {'term': {'owner_id': 'someuser'}}]}}

    def test_very_complicated_query(self):
        query_string = '((assignments.assignee_id:someuser OR assignments.is_completed:false) ' \
                       'OR (value:true AND another:false AND (some:true AND NOT field:true)) ' \
                       'AND NOT (complicated:true OR NOT complicated:false)) ' \
                       'OR owner_id:someuser AND NOT completed:false'
        result = compile_es_query(query_string)
        assert result == {'bool': {
            'must_not': [{'term': {'completed': 'false'}}],
            'should': [{'bool': {
                'must_not': [{'bool': {
                    'should_not': [{'term': {'complicated': 'true'}}, {'term': {'complicated': 'false'}}]}}],
                'should': [{'bool': {
                    'should': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                        'should': [{'term': {'assignments_nested.assignee_id': 'someuser'}},
                                   {'term': {'assignments_nested.is_completed': 'false'}}]}}}}]}},
                    {'bool': {
                        'must': [{'term': {'value': 'true'}},
                                       {'term': {'another': 'false'}},
                                       {'bool': {'must_not': [{'term': {'some': 'true'}}, {'term': {'field': 'true'}}]}}]}}]}},
                {'term': {'owner_id': 'someuser'}}]}}
