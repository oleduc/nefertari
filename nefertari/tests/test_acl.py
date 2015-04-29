import pytest
from mock import Mock
from pyramid.security import ALL_PERMISSIONS, Allow, Everyone, Authenticated

from nefertari import acl


class TestACLsUnit(object):

    def test_baseacl_init(self):
        acl_obj = acl.BaseACL(request='foo')
        assert acl_obj.request == 'foo'
        assert acl_obj.__acl__ == [(Allow, 'g:admin', ALL_PERMISSIONS)]
        assert acl_obj.__context_acl__ == [
            (Allow, 'g:admin', ALL_PERMISSIONS)]

    def test_baseacl_acl_getter(self):
        acl_obj = acl.BaseACL(request='foo')
        assert acl_obj.acl is acl_obj.__acl__
        assert acl_obj.acl == [(Allow, 'g:admin', ALL_PERMISSIONS)]

    def test_baseacl_acl_setter(self):
        acl_obj = acl.BaseACL(request='foo')
        assert acl_obj.acl == [(Allow, 'g:admin', ALL_PERMISSIONS)]
        ace = (Allow, Everyone, ['index', 'show'])
        with pytest.raises(AssertionError):
            acl_obj.acl = [ace]
        acl_obj.acl = ace
        assert acl_obj.acl == [(Allow, 'g:admin', ALL_PERMISSIONS), ace]

    def test_baseacl_context_acl(self):
        acl_obj = acl.BaseACL(request='foo')
        assert acl_obj.context_acl(None) is acl_obj.__context_acl__

    def test_baseacl_getitem_no_context_cls(self):
        acl_obj = acl.BaseACL(request='foo')
        assert acl_obj.__context_class__ is None
        with pytest.raises(AssertionError):
            acl_obj.__getitem__('foo')

    def test_baseacl_getitem(self):
        acl_obj = acl.BaseACL(request='foo')
        clx_cls = Mock()
        clx_cls.id_field.return_value = 'storyname'
        acl_obj.__context_class__ = clx_cls
        obj = acl_obj.__getitem__('foo')
        clx_cls.id_field.assert_called_once_with()
        clx_cls.get.assert_called_once_with(
            __raise=True, storyname='foo')
        assert obj.__acl__ == acl_obj.__context_acl__
        assert obj.__parent__ == acl_obj
        assert obj.__name__ == 'foo'

    def test_rootacl(self):
        acl_obj = acl.RootACL(request='foo')
        assert acl_obj.__acl__ == [(Allow, 'g:admin', ALL_PERMISSIONS)]
        assert acl_obj.request == 'foo'

    def test_adminacl(self):
        acl_obj = acl.AdminACL(request='foo')
        assert isinstance(acl_obj, acl.BaseACL)
        assert acl_obj['foo'] == 1
        assert acl_obj['qweoo'] == 1

    def test_guestacl_acl(self):
        acl_obj = acl.GuestACL(request='foo')
        assert acl_obj.acl == [
            (Allow, 'g:admin', ALL_PERMISSIONS),
            (Allow, Everyone, ['index', 'show'])
        ]

    def test_guestacl_context_acl(self):
        acl_obj = acl.GuestACL(request='foo')
        assert acl_obj.context_acl('asdasd') == [
            (Allow, 'g:admin', ALL_PERMISSIONS),
            (Allow, Everyone, ['index', 'show']),
        ]

    def test_authenticatedreadacl_acl(self):
        acl_obj = acl.AuthenticatedReadACL(request='foo')
        assert acl_obj.acl == [
            (Allow, 'g:admin', ALL_PERMISSIONS),
            (Allow, Authenticated, 'index')
        ]

    def test_authenticatedreadacl_context_acl(self):
        acl_obj = acl.AuthenticatedReadACL(request='foo')
        assert acl_obj.context_acl('asdasd') == [
            (Allow, 'g:admin', ALL_PERMISSIONS),
            (Allow, Authenticated, 'show'),
        ]
