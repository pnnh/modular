struct MyPair:
    var first: Int
    var second: Int

    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second

    fn __lt__(self, rhs: MyPair) -> Bool:
        return self.first < rhs.first or (
            self.first == rhs.first and self.second < rhs.second
        )


def pair_test() -> Bool:
    let p = MyPair(1, 2)
    return True


def main():
    print("Hello, world!")
    for x in range(9, 0, -3):
        print(x)
    print(pair_test())
