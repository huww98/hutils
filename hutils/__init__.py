class NoOp:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
