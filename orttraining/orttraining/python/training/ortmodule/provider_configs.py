
class ProviderConfigs:

    def __init__(self,
                 provider_type="",
                 backend="CPU",
                 ):
        self._provider_type = provider_type
        self._backend= backend

    @property
    def provider(self):
        return self._provider_type

    @property
    def backend(self):
        return self._backend