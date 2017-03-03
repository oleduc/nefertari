from mock import Mock
import pytest

from nefertari.es_query import compile_es_query, Tokenizer, Buffer, apply_analyzer, Node


class TestNode:

    def test_node_tree_first_level(self):
        tokens = ['(', 'token', ')']
        tree = Node.build_tree(tokens)
        assert tree == [['token']]

    def test_node_tree_wrong_level(self):
        tokens = ['(', 'token', ')', ')']
        with pytest.raises(ValueError):
            Node.build_tree(tokens)

    def test_node_tree_second_level(self):
        tokens = ['(','(', 'token', ')', 'token', '(', 'token', ')', ')']
        tree = Node.build_tree(tokens)
        assert tree == [[['token'], 'token', ['token']]]

    def test_node_tree_third_level(self):
        tokens = ['(', '(', '(', 'token', ')', ')', 'token', '(', 'token', ')', ')']
        tree = Node.build_tree(tokens)
        assert tree == [[[['token']], 'token', ['token']]]


class TestProcessors:
    pass


class TestESQueryCompilation(object):

    def test_build_parse_tokens(self):
        query_string = 'item:value OR item:value'
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(query_string)
        assert tokens == ['item:value', 'OR', 'item:value']
        query_string = '((item:value OR item:value) AND NOT (item:OR OR item:NOT)) OR true:true'
        tokens = tokenizer.tokenize(query_string)
        print(tokens)
        assert tokens == ['(', '(', 'item:value', 'OR', 'item:value', ')',
                          'AND NOT', '(', 'item:OR', 'OR', 'item:NOT',
                          ')', ')', 'OR', 'true:true']

    def test_remove_top_level_brackets_if_needless(self):
        query_string = '(item:value OR item:value)'
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(query_string)
        tree = Node.build_tree(tokens)
        assert tree == ['item:value', 'OR', 'item:value']

    def test_build_tree(self):
        query_string = '(item:value OR item:value) AND ((item:value OR item:value AND complicated:false) OR (item:value OR item:value))'
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(query_string)
        tree = Node.build_tree(tokens)
        assert tree == [['item:value', 'OR', 'item:value'], 'AND',
                        [['item:value', 'OR', 'item:value', 'AND', 'complicated:false'], 'OR',
                         ['item:value', 'OR', 'item:value']]]

    def test_nested_query(self):
        query_string = 'assignments.assignee_id:someuse'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}}]}}}}]}}

    def test_nested_query_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuse.user.@b.a.b.la."'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'query': {'bool': {
                'must': [{'term': {'assignments_nested.assignee_id': 'someuse.user.@b.a.b.la.'}}]}},
                'path': 'assignments_nested'}}]}}

    def test_nested_query_and_with_quotes(self):
        query_string = 'assignments.assignee_id:"someuser.some.last.name" ' \
                 'AND assignments.assignor_id:"changed.user.name"'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'path': 'assignments_nested', 'query': {
            'bool': {
                'must': [{'term': {'assignments_nested.assignee_id': 'someuser.some.last.name'}},
                         {'term': {'assignments_nested.assignor_id': 'changed.user.name'}}]}}}}]}}]}}

    def test_nested_query_and(self):
        query_string = 'assignments.assignee_id:someuse AND assignments.is_completed:true'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}},
                     {'term': {'assignments_nested.is_completed': 'true'}}]}},
                                                        'path': 'assignments_nested'}}]}}]}}

    def test_nested_query_complicated(self):
        query_string = 'assignments.assignee_id:someuse AND NOT assignments.assignor_id:someusesaqk AND assignments.is_completed:true'
        params = {'es_q': query_string}
        result = compile_es_query(params)

        assert result == {'bool': {'must': [{'bool': {
            'must_not': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                'must': [{'term': {'assignments_nested.assignor_id': 'someusesaqk'}}]}}}}],
            'must': [{'nested': {'path': 'assignments_nested', 'query': {
                'bool': {
                    'must': [
                        {'term': {'assignments_nested.assignee_id': 'someuse'}},
                        {'term': {
                            'assignments_nested.is_completed': 'true'}}]}}}}]}}]}}

    def test_nested_query_inside_query(self):
        query_string = '(assignments.assignee_id:someuser OR assignments.is_completed:false AND assignments.assignor_id:another) OR owner_id:someuser'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'bool': {
                'should': [{'bool': {
                    'should': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                        'should': [{'term': {'assignments_nested.assignee_id': 'someuser'}},
                                   {'term': {'assignments_nested.is_completed': 'false'}}],
                        'minimum_should_match': 1}}}}], 'must': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                        'must': [{'term': {'assignments_nested.assignor_id': 'another'}}]}}}}],
                    'minimum_should_match': 1}}, {'term': {'owner_id': 'someuser'}}], 'minimum_should_match': 1}}]}}

    def test_very_complicated_query(self):
        query_string = '((assignments.assignee_id:someuser OR assignments.is_completed:false) ' \
                       'OR (value:true AND another:false AND (some:true AND NOT field:true)) ' \
                       'AND NOT (complicated:true OR complicated:false)) ' \
                       'OR owner_id:someuser AND NOT completed:false'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'bool': {
                'must_not': [{'term': {'completed': 'false'}}],
                'should': [{'bool': {
                    'must_not': [{'bool': {
                        'should': [{'term': {'complicated': 'true'}},
                                   {'term': {'complicated': 'false'}}],
                        'minimum_should_match': 1}}],
                    'should': [{'bool': {
                        'should': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                            'should': [{'term': {'assignments_nested.assignee_id': 'someuser'}},
                                       {'term': {'assignments_nested.is_completed': 'false'}}],
                            'minimum_should_match': 1}}}}],
                        'minimum_should_match': 1}},
                        {'bool': {'must': [{'term': {'value': 'true'}},
                                           {'term': {'another': 'false'}}, {'bool': {
                                'must': [{'term': {'some': 'true'}}],
                                'must_not': [{'term': {'field': 'true'}}]}}]}}],
                    'minimum_should_match': 1}}, {'term': {'owner_id': 'someuser'}}],
                'minimum_should_match': 1}}]}}

    def test_range_query_nested(self):
        query_string = 'schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] AND schedules.obj_status:active'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'bool': {
                'must': [{'nested': {'query': {'bool': {
                    'must': [{'range': {'schedules_nested.end_date': {
                        'lte': '2016-10-18T02:59:59',
                        'gte': '2016-10-11T03:00:00'}}},
                        {'term': {'schedules_nested.obj_status': 'active'}}]}},
                    'path': 'schedules_nested'}}]}}]}}

    def test_range_query(self):
        query_string = '(x:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] OR z:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59]) AND (schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59] AND schedules.obj_status:active)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {
            'must': [{
                'bool': {
                    'should': [{
                        'range': {'x': {
                            'gte': '2016-10-11T03:00:00',
                            'lte': '2016-10-18T02:59:59'}}}, {
                        'range': {'z': {
                            'gte': '2016-10-11T03:00:00',
                            'lte': '2016-10-18T02:59:59'}}}],
                    'minimum_should_match': 1}}, {
                'bool': {
                    'must': [{
                        'nested': {'query': {'bool': {
                            'must': [{
                                'range': {'schedules_nested.end_date': {
                                    'gte': '2016-10-11T03:00:00',
                                    'lte': '2016-10-18T02:59:59'}}},
                                {'term': {'schedules_nested.obj_status': 'active'}}]}},
                            'path': 'schedules_nested'}}]}}]}}]}}

    def test_range_query_with_missed_from_and_to(self):
        query_string = '(x:[_missing_ TO 2016-10-18T02:59:59] OR z:[_missing_ TO 2016-10-18T02:59:59]) AND (schedules.end_date:[2016-10-11T03:00:00 TO _missing_] AND schedules.obj_status:active)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {
            'must': [{
                'bool': {
                    'should': [{
                        'range': {'x': {
                            'lte': '2016-10-18T02:59:59'}}}, {
                        'range': {'z': {
                            'lte': '2016-10-18T02:59:59'}}}],
                'minimum_should_match': 1
                }}, {
                'bool': {
                    'must': [{
                        'nested': {'query': {'bool': {
                            'must': [{
                                'range': {'schedules_nested.end_date': {
                                    'gte': '2016-10-11T03:00:00'}}},
                                {'term': {'schedules_nested.obj_status': 'active'}}]}},
                            'path': 'schedules_nested'}}]}}]}}]}}

    def test_range_query_with_brackets(self):
        query_string ='((schedules.end_date:[2016-10-11T03:00:00 TO 2016-10-18T02:59:59]))'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'query': {'bool': {
                'must': [{
                    'range': {'schedules_nested.end_date': {'gte': '2016-10-11T03:00:00',
                                                            'lte': '2016-10-18T02:59:59'}}}]}},
                'path': 'schedules_nested'}}]}}

    def test_statements_in_parentheses(self):
        query_string = '(assignments.assignee_id:someuse) AND (assignments.is_completed:true)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}},
                     {'term': {'assignments_nested.is_completed': 'true'}}]}},
                                                        'path': 'assignments_nested'}}]}}]}}

    def test_do_not_apply_needless_parentheses(self):
        query_string = '((((assignments.assignee_id:someuse)) AND ((assignments.is_completed:true))))'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}},
                     {'term': {'assignments_nested.is_completed': 'true'}}]}},
                                                        'path': 'assignments_nested'}}]}}]}}

    def test_do_not_apply_needless_parentheses_with_statements(self):
        query_string = '((((assignments.assignee_id:someuse)) AND ((assignments.is_completed:true))) OR assignments.is_completed:brayan)'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'bool': {
                'must': [{'nested': {'query': {'bool': {
                    'must': [{'term': {'assignments_nested.assignee_id': 'someuse'}}]}},
                    'path': 'assignments_nested'}}, {'bool': {
                    'should': [{'nested': {'query': {'bool': {
                        'should': [{'term': {'assignments_nested.is_completed': 'true'}},
                                   {'term': {'assignments_nested.is_completed': 'brayan'}}],
                        'minimum_should_match': 1
                    }},
                        'path': 'assignments_nested'}}], 'minimum_should_match': 1}}]}}]}}

    def test_simple_query_with_parentheses(self):
        query_string = '(assignments.assignee_id:"qweqweqwe")'
        params = {'es_q': query_string}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'nested': {'path': 'assignments_nested', 'query': {'bool': {
                'must': [{'term': {'assignments_nested.assignee_id': 'qweqweqwe'}}]}}}}]}}

    def test_apply_custom_analyzer(self):
        document_cls = Mock()
        document_cls.get_es_mapping = Mock(return_value=({'Assignment': {
            'properties': {
                'assignee_id': {'type': 'string', 'analyzer': 'email'},
                'simple': {'type': 'string'}}}}, []))

        get_document_cls = Mock(return_value=document_cls)
        params = {'assignee_id': 'some_user', 'simple': 'new_value'}
        result = apply_analyzer(params, 'Assignment', get_document_cls)
        assert result == {'bool': {'must': [{'term': {'assignee_id': 'some_user'}}]}}

    def test_apply_custom_analyzer_doesnt_duplicated(self):

        def get_document_cls(model_name):
            document = Mock()

            if model_name == 'Assignment':
                document.get_es_mapping = Mock(return_value=({'Assignment': {
                    'properties': {
                        'assignee_id': {'type': 'string', 'analyzer': 'email'},
                        'simple': {'type': 'string'}}}}, []))
                return document
            if model_name == 'Task':
                document.get_es_mapping = Mock(return_value=({'Task': {
                    'properties': {
                        'assignee_id': {'type': 'string', 'analyzer': 'email'},
                        'simple': {'type': 'string'}}}}, []))

                return document
        params = {'assignee_id': 'some_user', 'simple': 'new_value'}
        result = apply_analyzer(params, 'Assignment,Task', get_document_cls)
        assert result == {'bool': {'must': [{'term': {'assignee_id': 'some_user'}}]}}

    def test_parse_statement_inside_param_value(self):
        query_string = '(assignments.assignee_id:"qweqweqwe")'
        params = {'es_q': query_string, 'project_id': '(2 OR 3)'}
        result = compile_es_query(params)
        assert result == {'bool': {
            'must': [{'bool': {
                'must': [{'nested': {'path': 'assignments_nested',
                                     'query': {'bool': {'must': [{'term': {'assignments_nested.assignee_id': 'qweqweqwe'}}]}}}},
                         {'bool': {'should': [{'term': {'project_id': '2'}}, {'term': {'project_id': '3'}}],
                                   'minimum_should_match': 1}}]}}]}}

    def test_apply_boost(self):
        query_string = '(assignments.assignee_id:someuse) AND (assignments.is_completed:true)'
        params = {'es_q': query_string, '_boost': 'assignments.assignee_id:5,assignments.is_completed:10'}
        result = compile_es_query(params)
        print(result)
        assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'query': {'bool': {
            'must': [{'term': {'assignments_nested.assignee_id': 'someuse', 'boost': 5}},
                     {'term': {'assignments_nested.is_completed': 'true', 'boost': 10}}]}},
                                                        'path': 'assignments_nested'}}]}}]}}

    # def test_spaces(self):
    #     query_string = '(assignments.assignee_id:some   use) AND (assignments.is_completed:true)'
    #     params = {'es_q': query_string, '_boost': 'assignments.assignee_id:5,assignments.is_completed:10'}
    #     result = compile_es_query(params)
    #     print(result)
    #     assert result == {'bool': {'must': [{'bool': {'must': [{'nested': {'query': {'bool': {
    #         'must': [{'match_phrase': {'assignments_nested.assignee_id': 'some   use', 'boost': 5}},
    #                  {'term': {'assignments_nested.is_completed': 'true', 'boost': 10}}]}},
    #                                                     'path': 'assignments_nested'}}]}}]}}
    #
    # def test_empty_query(self):
    #     query_string = 'some'
    #     params = {'es_q': query_string, '_boost': 'assignments.assignee:2'}
    #     result = compile_es_query(params)
