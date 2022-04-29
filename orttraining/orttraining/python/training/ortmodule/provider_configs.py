class ProviderConfigs:
    def __init__(
        self,
        provider="",
        backend="CPU",
        precision="FP32",
    ):
        self._provider = provider
        self._backend = backend
        self._precision = precision

    @property
    def provider(self):
        return self._provider

    @property
    def backend(self):
        return self._backend

    @property
    def precision(self):
        return self._precision
