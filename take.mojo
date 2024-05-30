struct FileDescriptor:
    var fd: Int

    fn __takeinit__(inout self, inout existing: Self):
        self.fd = existing.fd
        existing.fd = -1

    fn __moveinit__(inout self, owned existing: Self):
        self.fd = existing.fd
        existing.fd = -1

    fn __init__(inout self, fd: Int):
        self.fd = fd

    # fn __init__(inout self, path: String):
    #     self.fd = open(path, O_RDONLY)

    # fn __del__(owned self):
    #     if self.fd != -1:
    #         close(self.fd)
