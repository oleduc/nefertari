import json
import logging
from datetime import date, datetime

from nefertari import wrappers
from nefertari.json_httpexceptions import JHTTPOk, JHTTPCreated

log = logging.getLogger(__name__)


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")  # iso

        try:
            return super(_JSONEncoder, self).default(obj)
        except TypeError:
            return str(obj)  # fallback to str


class JsonRendererFactory(object):

    def __init__(self, info):
        """ Constructor: info will be an object having the
        following attributes: name (the renderer name), package
        (the package that was 'current' at the time the
        renderer was registered), type (the renderer type
        name), registry (the current application registry) and
        settings (the deployment settings dictionary). """
        pass

    def _set_content_type(self, system):
        request = system.get('request')
        if request:
            response = request.response
            ct = response.content_type
            if ct == response.default_content_type:
                response.content_type = 'application/json'

    def _render_response(self, value, system):
        view = system['view']
        enc_class = getattr(
            view, '_json_encoder', _JSONEncoder) or _JSONEncoder
        return json.dumps(value, cls=enc_class)

    def __call__(self, value, system):
        """ Call the renderer implementation with the value
        and the system value passed in as arguments and return
        the result (a string or unicode object). The value is
        the return value of a view.  The system value is a
        dictionary containing available system values
        (e.g. view, context, and request).
        """
        self._set_content_type(system)
        # run after_calls on the value before jsonifying
        value = self.run_after_calls(value, system)
        return self._render_response(value, system)

    def run_after_calls(self, value, system):
        request = system.get('request')
        if request and hasattr(request, 'action'):

            if request.action in ['index', 'show']:
                value = wrappers.wrap_in_dict(request)(result=value)

        return value


class DefaultResponseRendererMixin(object):
    def _get_common_kwargs(self, system):
        request = system['request']
        enc_class = getattr(
            system['view'], '_json_encoder', _JSONEncoder) or _JSONEncoder
        return {
            'request': request,
            'encoder': enc_class,
        }

    def render_create(self, value, system, common_kw):
        kw = common_kw.copy()
        kw['resource'] = value
        if hasattr(value, 'to_dict'):
            kw['resource'] = value.to_dict()
            resource = system['view']._resource
            id_name = resource.id_name
            obj_id = getattr(value, value.pk_field())
            kw['location'] = system['request'].route_url(
                resource.uid, **{id_name: obj_id})
        return JHTTPCreated(**kw)

    def render_update(self, value, system, common_kw):
        kw = common_kw.copy()
        if hasattr(value, 'to_dict'):
            resource = system['view']._resource
            id_name = resource.id_name
            obj_id = getattr(value, value.pk_field())
            kw['location'] = system['request'].route_url(
                resource.uid, **{id_name: obj_id})
        return JHTTPOk(**kw)

    def render_delete(self, value, system, common_kw):
        return JHTTPOk("Deleted", common_kw.copy())

    def render_delete_many(self, value, system, common_kw):
        msg = "Deleted {} {}(s) objects".format(
            value, system['view'].Model.__name__)
        return JHTTPOk(msg, common_kw.copy())

    def render_update_many(self, value, system, common_kw):
        msg = "Updated {} {}(s) objects".format(
            value, system['view'].Model.__name__)
        return JHTTPOk(msg, common_kw.copy())

    def _render_response(self, value, system):
        method_name = 'render_{}'.format(system['request'].action)
        method = getattr(self, method_name, None)
        if method is not None:
            common_kw = self._get_common_kwargs(system)
            return method(value, system, common_kw).body
        return super(DefaultResponseRendererMixin, self)._render_response(
            value, system)


class NefertariJsonRendererFactory(DefaultResponseRendererMixin,
                                   JsonRendererFactory):
    """ Special json renderer which will apply all after_calls(filters)
    to the result.
    """
    def run_after_calls(self, value, system):
        request = system.get('request')
        if request and hasattr(request, 'action'):
            after_calls = getattr(request, 'filters', {})
            for call in after_calls.get(request.action, []):
                value = call(**dict(request=request, result=value))
        return value
