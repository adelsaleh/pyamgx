import pytest

import pyamgx


def test_check_error_preserves_return_code():
    with pytest.raises(pyamgx.AMGXError) as raised:
        pyamgx.check_error(pyamgx.RC.NO_MEMORY)

    assert raised.value.error_code == pyamgx.RC.NO_MEMORY
    assert str(raised.value) == pyamgx.get_error_string(pyamgx.RC.NO_MEMORY)
