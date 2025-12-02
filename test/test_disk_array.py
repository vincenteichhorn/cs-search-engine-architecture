import os
import pytest
from sea.disk_array import DiskArray


def test_disk_array_in_memory(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Hello, World!"
    data2 = b"Testing DiskArray"

    id = disk_array.py_append(data1)
    assert id == 0
    id = disk_array.py_append(data2)
    assert id == 1

    retrieved1 = disk_array.py_get(0)
    retrieved2 = disk_array.py_get(1)

    assert retrieved1 == data1
    assert retrieved2 == data2


def test_disk_array_on_disk(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Data on disk 1"
    data2 = b"Data on disk 2"

    id = disk_array.py_append(data1)
    assert id == 0
    id = disk_array.py_append(data2)
    assert id == 1

    disk_array.flush()

    with open(os.path.join(tmp_path, "data.dat"), "rb") as f:
        disk_data = f.read()
        print(disk_data)  # print the raw data on disk

    retrieved1 = disk_array.py_get(0)
    print(retrieved1)  # print the retrieved data right
    retrieved2 = disk_array.py_get(1)
    print(retrieved2)  # this is not the correct data

    assert retrieved1 == data1
    assert retrieved2 == data2


def test_disk_array_on_disk_and_memory(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Mixed storage 1"
    data2 = b"Mixed storage 2"
    data3 = b"Mixed storage 3"

    id = disk_array.py_append(data1)
    assert id == 0
    disk_array.flush()  # Force first entry to disk
    id = disk_array.py_append(data2)
    assert id == 1
    id = disk_array.py_append(data3)
    assert id == 2

    retrieved1 = disk_array.py_get(0)
    retrieved2 = disk_array.py_get(1)
    retrieved3 = disk_array.py_get(2)

    assert retrieved1 == data1
    assert retrieved2 == data2
    assert retrieved3 == data3


def test_disk_array_out_of_bounds(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data = b"Out of bounds test"
    id = disk_array.py_append(data)
    assert id == 0

    with pytest.raises(IndexError):
        disk_array.py_get(1)  # Accessing out of bounds
