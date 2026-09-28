"""Entity-table contract: code mapping matches database/main_database.db."""

from msdl_jci.config.settings import get_settings
from msdl_jci.utils.data_loader import ENTITY_TABLES, read_sqlite_table


def test_entity_tables_match_db():
    db = get_settings().SQLITE_DB_PATH
    assert set(ENTITY_TABLES) == {
        "jci_historical",
        "bi_rate",
        "inflation_data",
        "kurs_usdidr",
        "cnbc_ihsg_articles",
        "detik_ihsg_articles",
        "kontan_ihsg_articles",
    }
    for table, cols in ENTITY_TABLES.items():
        df = read_sqlite_table(table, db)  # raises on schema drift
        assert len(df) > 0
        assert all(c in df.columns for c in cols)
