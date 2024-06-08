from python import Python as py

fn main() raises:
    var np = py.import_module("numpy")

    var array = np.array([1, 2, 3])
    print(array)
