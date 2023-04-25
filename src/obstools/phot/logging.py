

class TqdmStreamAdapter:

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def __enter__(self):
        pass

    def __exit__(self):
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr

    def __eq__(self, other):
        return other is self.stream

    def write(self, msg):
        tqdm.write(msg, self.stream, end='')

    def flush(self):
        pass
        # return self.stream.flush()


class TqdmLogAdapter:
    def __init__(self, sink_ids=()):
        sink_ids = list(sink_ids)
        if not sink_ids:
            sink_ids = [
                id_
                for id_, handler in logger._core.handlers.items()
                if isinstance(handler._sink, StreamSink)
            ]

        self.streams = {
            id_: logger._core.handlers[id_]._sink._stream
            for id_ in sink_ids
        }

    def __enter__(self):
        for sink in self.sinks.values():
            sink._stream = TqdmStreamAdapter(sink._stream)

    def __exit__(self):
        for id_, sink in self.sinks.items():
            sink._stream = sink._stream.stream

