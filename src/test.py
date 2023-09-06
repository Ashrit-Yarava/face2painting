def func2(func):
    def inner():
        func()
    return inner()


@func2
def func1():
    print("World!")


func1()
