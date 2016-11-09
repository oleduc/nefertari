from nefertari.es_query import compile_es_query, _get_tokens, _build_tree, _attach_nested


class TestESQueryCompilation(object):

    def test_build_parse_tokens(self):
        query_string = 'item:value OR item:value'
        assert ['item:value', 'OR', 'item:value'] == _get_tokens(query_string)
        query_string = '((item:value OR item:value) AND NOT (item:OR OR NOT item:NOT)) OR true:true'
        assert _get_tokens(query_string) == ['(', '(', 'item:value', 'OR', 'item:value', ')',
                                             'AND NOT', '(', 'item:OR', 'OR NOT', 'item:NOT',
                                             ')', ')', 'OR', 'true:true']

    def test_remove_top_level_brackets_if_needless(self):
        query_string = '(item:value OR item:value)'
        tokens = _get_tokens(query_string)
        tree = _build_tree(tokens)
        assert tree == ['item:value', 'OR', 'item:value']

    def test_build_tree(self):
        query_string = '(item:value OR item:value) AND ((item:value OR item:value AND complicated:false) OR (item:value OR item:value))'
        tokens = _get_tokens(query_string)
        tree = _build_tree(tokens)
        assert tree == [['item:value', 'OR', 'item:value'], 'AND',
                        [['item:value', 'OR', 'item:value', 'AND', 'complicated:false'], 'OR',
                         ['item:value', 'OR', 'item:value']]]

    def test_nested_query(self):
        query_string = 'assignments.assignee_id:someuse'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{
                'nested': {'query': {'bool': {
                    'must': [{'match': {'assignments_nested.assignee_id': 'someuse'}}]}},
                    'path': 'assignments_nested'}}]}}

    def test_nested_query_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuse.user.@b.a.b.la."'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                'must': [{'match': {'assignments_nested.assignee_id': 'someuse.user.@b.a.b.la.'}}]}}}}]}}


    def test_nested_query_and_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuser.some.last.name" ' \
                 'AND assignments.assignor_id:"changed.user.name"'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'nested': {'path': 'assignments_nested', 'query': {
            'bool': {
                'must': [{'match': {'assignments_nested.assignee_id': 'someuser.some.last.name'}},
                         {'match': {'assignments_nested.assignor_id': 'changed.user.name'}}]}}}}]}}

    def test_nested_query_and(self):
        query_string = 'assignments.assignee_id:someuse AND assignments.is_completed:true'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'match': {'assignments_nested.assignee_id': 'someuse'}},
                     {'match': {'assignments_nested.is_completed': 'true'}}]}},
                                                        'path': 'assignments_nested'}}]}}

    def test_nested_query_complicated(self):
        query_string = 'assignments.assignee_id:someuse AND NOT assignments.assignor_id:someusesaqk AND assignments.is_completed:true'
        params = {'es_q': query_string}
        result = compile_es_query(params)

        assert result == {'bool': {
            'must_not': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                'must': [{'match': {'assignments_nested.assignor_id': 'someusesaqk'}}]}}}}],
            'must': [{'nested': {'path': 'assignments_nested', 'query': {
                'bool': {
                    'must': [
                        {'match': {'assignments_nested.assignee_id': 'someuse'}},
                        {'match': {
                            'assignments_nested.is_completed': 'true'}}]}}}}]}}

    def test_nested_query_inside_query(self):
        query_string = '(assignments.assignee_id:someuser OR assignments.is_completed:false AND assignments.assignor_id:another) OR owner_id:someuser'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'should': [{'bool': {
                'must': [
                    {'nested': {'query': {'bool': {
                        'must': [{'match': {'assignments_nested.assignor_id': 'another'}}]}},
                                'path': 'assignments_nested'}}],
                'should': [
                    {'nested': {'query': {'bool': {
                        'should': [{'match': {'assignments_nested.assignee_id': 'someuser'}},
                                   {'match': {'assignments_nested.is_completed': 'false'}}]}},
                        'path': 'assignments_nested'}}]
            }},
                {'match': {'owner_id': 'someuser'}}]}}

    def test_very_complicated_query(self):
        query_string = '((assignments.assignee_id:someuser OR assignments.is_completed:false) ' \
                       'OR (value:true AND another:false AND (some:true AND NOT field:true)) ' \
                       'AND NOT (complicated:true OR NOT complicated:false)) ' \
                       'OR owner_id:someuser AND NOT completed:false'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'should':
                [{
                    'bool':
                        {
                            'should': [{'bool':
                                {
                                    'should': [{'nested': {'query': {'bool': {
                                        'should': [{'match': {
                                            'assignments_nested.assignee_id': 'someuser'}},
                                                   {'match': {
                                                       'assignments_nested.is_completed': 'false'}}]}},
                                        'path': 'assignments_nested'}}]}},
                                {'bool':
                                    {
                                        'must': [{'match': {'value': 'true'}},
                                                 {'match': {'another': 'false'}},
                                                 {'bool': {
                                                     'must_not': [{'match': {'field': 'true'}}],

                                                     'must': [{'match': {'some': 'true'}}]}}]}}],
                            'must_not': [{'bool': {
                                'should': [{'match': {'complicated': 'true'}}],
                                'should_not': [{'match': {'complicated': 'false'}}]}}]}},
                    {'match': {'owner_id': 'someuser'}}],
            'must_not': [{'match': {'completed': 'false'}}]}}

    def test_range_query_nested(self):
        query_string = 'schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] AND schedules.obj_status:active'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'query': {'bool': {
                'must': [{
                    'range': {'schedules_nested.end_date': {
                        'lte': '2016-10-18T02:59:59',
                        'gte': '2016-10-11T03:00:00'}}},
                    {
                        'match': {'schedules_nested.obj_status': 'active'}}]}},
                'path': 'schedules_nested'}}]}}

    def test_range_query(self):
        query_string = '(x:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] OR z:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59]) AND (schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] AND schedules.obj_status:active)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{
                'bool': {
                    'should': [{
                        'range': {'x': {
                            'gte': '2016-10-11T03:00:00',
                            'lte': '2016-10-18T02:59:59'}}}, {
                        'range': {'z': {
                            'gte': '2016-10-11T03:00:00',
                            'lte': '2016-10-18T02:59:59'}}}]}}, {
                'bool': {
                    'must': [{
                        'nested': {'query': {'bool': {
                            'must': [{
                                'range': {'schedules_nested.end_date': {
                                    'gte': '2016-10-11T03:00:00',
                                    'lte': '2016-10-18T02:59:59'}}},
                                {'match': {'schedules_nested.obj_status': 'active'}}]}},
                            'path': 'schedules_nested'}}]}}]}}

    def test_range_query_with_missed_from_and_to(self):
        query_string = '(x:[_missing_ TO 2016-10-18T02:59:59] OR z:[_missing_ TO 2016-10-18T02:59:59]) AND (schedules.end_date:[2016-10-11T03:00:00 TO _missing_] AND schedules.obj_status:active)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{
                'bool': {
                    'should': [{
                        'range': {'x': {
                            'lte': '2016-10-18T02:59:59'}}}, {
                        'range': {'z': {
                            'lte': '2016-10-18T02:59:59'}}}]}}, {
                'bool': {
                    'must': [{
                        'nested': {'query': {'bool': {
                            'must': [{
                                'range': {'schedules_nested.end_date': {
                                    'gte': '2016-10-11T03:00:00'}}},
                                {'match': {'schedules_nested.obj_status': 'active'}}]}},
                            'path': 'schedules_nested'}}]}}]}}

    def test_range_query_with_brackets(self):
        query_string ='schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59]'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'path': 'schedules_nested', 'query': {'bool': {
                'must': [{'range': {'schedules_nested.end_date': {'lte': '2016-10-18T02:59:59',
                                                                  'gte': '2016-10-11T03:00:00'
                                                                  }}}]}}}}]}}

    def test_array_matching(self):
        query_string = 'inbox:["some@user.com", "another@user.com"]'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == ''
