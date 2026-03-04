from connectors.xls_data_source import XLSDataSource
import pytest


def test_read_xls_file_parses_xml_correctly(tmp_path):
    source = XLSDataSource()

    xml_content = """
    <Workbook>
        <Worksheet>
            <Table>
                <Row>
                    <Cell><Data>Ignore1</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>Col A</Data></Cell>
                    <Cell><Data>Col B (%)</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>1</Data></Cell>
                    <Cell><Data>2</Data></Cell>
                </Row>
            </Table>
        </Worksheet>

        <Worksheet>
            <Table>
                <Row>
                    <Cell><Data>X</Data></Cell>
                    <Cell><Data>Y</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>9</Data></Cell>
                    <Cell><Data>8</Data></Cell>
                </Row>
            </Table>
        </Worksheet>
    </Workbook>
    """

    file_path = tmp_path / "testfile.xml"
    file_path.write_text(xml_content)

    df = source.read_xls_file(file_path=str(file_path), sheet_number=0, skiprows=1)

    assert list(df.columns) == ["COL_A", "COL_B"]
    assert df.iloc[0]["COL_A"] == "1"
    assert df.iloc[0]["COL_B"] == "2"
    assert source.file_path == str(file_path)
    assert source.dataframe is not None


def test_read_xls_file_selects_correct_sheet(tmp_path):
    source = XLSDataSource()

    xml_content = """
    <Workbook>
        <Worksheet>
            <Table>
                <Row><Cell><Data>A</Data></Cell></Row>
                <Row><Cell><Data>1</Data></Cell></Row>
            </Table>
        </Worksheet>

        <Worksheet>
            <Table>
                <Row><Cell><Data>Header1</Data></Cell></Row>
                <Row><Cell><Data>Value1</Data></Cell></Row>
            </Table>
        </Worksheet>
    </Workbook>
    """

    file_path = tmp_path / "testfile.xml"
    file_path.write_text(xml_content)

    df = source.set_file_path(str(file_path)).read(sheet_number=1, skiprows=0)

    assert list(df.columns) == ["HEADER1"]
    assert df.iloc[0]["HEADER1"] == "Value1"


def test_read_requires_file_path():
    source = XLSDataSource()
    with pytest.raises(ValueError, match="No XLS file path is set"):
        source.read()
