
class ProviderConfigs:

    def __init__(self,
                 provider="",
                 backend="CPU",
                 ):
        self._provider = provider
        self._backend= backend

    @property
    def provider(self):
        return self._provider

    @property
    def backend(self):
        return self._backend