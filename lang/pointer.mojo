from memory.unsafe import Pointer


struct HeapArray:
    var data: Pointer[Int]
    var size: Int

    fn __init__(inout self, size: Int, val: Int):
        self.size = size
        self.data = Pointer[Int].alloc(size)
        for i in range(self.size):
            self.data.store(i, val)

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.data = Pointer[Int].alloc(self.size)
        for i in range(self.size):
            self.data.store(i, other.data.load(i))

    fn __del__(owned self):
        self.data.free()

    fn dump(self):
        print_no_newline("[")
        for i in range(self.size):
            if i > 0:
                print_no_newline(", ")
            print_no_newline(self.data.load(i))
        print("]")


def main():
    let a = HeapArray(3, 1)
    a.dump()

    let b = HeapArray(4, 2)
    b.dump()
    a.dump()

    let c = a  # 如果未实现__copyinit__，将会报错
    c.dump()
