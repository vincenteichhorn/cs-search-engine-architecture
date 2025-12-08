import os
import shutil
import pytest
from sea.util.disk_array import DiskArray


def test_disk_array_in_memory(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Hello, World!"
    data2 = b"Testing DiskArray"

    payload_1 = 123
    id = disk_array.py_append(payload_1, data1)
    assert id == 0
    payload_2 = 456
    id = disk_array.py_append(payload_2, data2)
    assert id == 1

    print("appended")

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload_1
    assert retrieved["data"] == data1
    retrieved = disk_array.py_get(1)
    assert retrieved["payload"] == payload_2
    assert retrieved["data"] == data2


def test_disk_array_on_disk(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Data on disk 1"
    data2 = b"Data on disk 2"

    payload_1 = 123
    id = disk_array.py_append(payload_1, data1)
    assert id == 0
    payload_2 = 456
    id = disk_array.py_append(payload_2, data2)
    assert id == 1
    disk_array.flush()

    with open(os.path.join(tmp_path, "data.dat"), "rb") as f:
        disk_data = f.read()
        print(disk_data)  # print the raw data on disk

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload_1
    assert retrieved["data"] == data1
    retrieved = disk_array.py_get(1)
    assert retrieved["payload"] == payload_2
    assert retrieved["data"] == data2


def test_disk_array_on_disk_and_memory(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Mixed storage 1"
    data2 = b"Mixed storage 2"
    data3 = b"Mixed storage 3"

    payload1 = 123
    payload2 = 456
    payload3 = 789
    id = disk_array.py_append(payload1, data1)
    assert id == 0
    disk_array.flush()  # Force first entry to disk
    id = disk_array.py_append(payload2, data2)
    assert id == 1
    id = disk_array.py_append(payload3, data3)
    assert id == 2

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload1
    assert retrieved["data"] == data1
    retrieved = disk_array.py_get(1)
    assert retrieved["payload"] == payload2
    assert retrieved["data"] == data2
    retrieved = disk_array.py_get(2)
    assert retrieved["payload"] == payload3
    assert retrieved["data"] == data3


def test_disk_array_out_of_bounds(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data = b"Out of bounds test"
    id = disk_array.py_append(0, data)
    assert id == 0

    with pytest.raises(IndexError):
        disk_array.py_get(1)  # Accessing out of bounds


def test_disk_array_size(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    assert disk_array.size() == 0

    data = b"Size test"
    for i in range(5):
        disk_array.py_append(i, data)
        assert disk_array.size() == i + 1


def test_disk_array_append_to_last(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b"Part 1, "
    data2 = b"Part 2."

    payload = 999
    id = disk_array.py_append(payload, data1)
    assert id == 0

    # Append to the last entry
    id = disk_array.py_add_to_last(data2)
    assert id == 0
    assert disk_array.size() == 1

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload
    assert retrieved["data"] == data1 + data2


def test_disk_array_append_to_last_empty(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data = b"Partial data."
    # Append to the last entry when no entries exist
    id = disk_array.py_add_to_last(data)
    assert id == 0
    assert disk_array.size() == 1


def test_disk_array_append_to_zero_length_data(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path)

    data1 = b""
    data2 = b"Non-empty data."

    payload1 = 111
    id = disk_array.py_append(payload1, data1)
    assert id == 0

    payload2 = 222
    id = disk_array.py_add_to_last(data2)
    assert id == 0
    assert disk_array.size() == 1

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload1
    assert retrieved["data"] == data1 + data2


def test_disk_array_with_mmap(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("disk_array_test"))
    disk_array = DiskArray(tmp_path, open_read_maps=True)

    data1 = b"Memory-mapped 1"
    data2 = b"Memory-mapped 2"

    payload_1 = 111
    id = disk_array.py_append(payload_1, data1)
    assert id == 0
    payload_2 = 222
    id = disk_array.py_append(payload_2, data2)
    assert id == 1

    retrieved = disk_array.py_get(0)
    assert retrieved["payload"] == payload_1
    assert retrieved["data"] == data1
    retrieved = disk_array.py_get(1)
    assert retrieved["payload"] == payload_2
    assert retrieved["data"] == data2
