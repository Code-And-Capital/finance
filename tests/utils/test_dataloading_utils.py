from utils.dataloading_utils import read_xls_file


def test_read_xls_file_parses_xml_correctly(tmp_path):
    # --- Arrange ---
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

    # --- Act ---
    df = read_xls_file(file_path=str(file_path), sheet_number=0, skiprows=1)

    # --- Assert ---
    # Expected columns after normalization:
    # "COL A"  -> "COL_A"
    # "COL B (%)" -> "COL_B"
    assert list(df.columns) == ["COL_A", "COL_B"]

    # Expected data
    assert df.iloc[0]["COL_A"] == "1"
    assert df.iloc[0]["COL_B"] == "2"


def test_read_xls_file_selects_correct_sheet(tmp_path):
    # --- Arrange ---
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

    # --- Act ---
    df = read_xls_file(str(file_path), sheet_number=1, skiprows=0)

    # --- Assert ---
    assert list(df.columns) == ["HEADER1"]
    assert df.iloc[0]["HEADER1"] == "Value1"
