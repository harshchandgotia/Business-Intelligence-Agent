import re
from src.models.schema import DatabaseSchema


class TemplateEngine:
    """Deterministic SQL for simple queries. No LLM calls."""

    PATTERNS = {
        "total": {
            "regex": r"(total|sum)\s+(of\s+)?(\w+)",
            "template": 'SELECT SUM("{col}") AS total_{col} FROM "{table}"',
        },
        "count": {
            "regex": r"(how many|count)\s+(of\s+)?(\w+)",
            "template": 'SELECT COUNT(*) AS count FROM "{table}"',
        },
        "top_n_by": {
            "regex": r"top\s+(\d+)\s+(\w+)\s+by\s+(\w+)",
            "template": 'SELECT "{group}", SUM("{metric}") AS total '
                        'FROM "{table}" GROUP BY "{group}" '
                        'ORDER BY total DESC LIMIT {n}',
        },
        "top_n_in": {
            "regex": r"top\s+(\d+)\s+(\w+)\s+(?:in|from|of)\s+(?:the\s+)?(\w+)",
            "template": 'SELECT "{group}", COUNT(*) AS count '
                        'FROM "{table}" GROUP BY "{group}" '
                        'ORDER BY count DESC LIMIT {n}',
        },
        "top_n_simple": {
            "regex": r"top\s+(\d+)\s+(\w+)",
            "template": 'SELECT "{group}", COUNT(*) AS count '
                        'FROM "{table}" GROUP BY "{group}" '
                        'ORDER BY count DESC LIMIT {n}',
        },
        "average": {
            "regex": r"(average|avg)\s+(of\s+)?(\w+)",
            "template": 'SELECT AVG("{col}") AS avg_{col} FROM "{table}"',
        },
    }

    def try_match(self, query: str, schema: DatabaseSchema) -> str | None:
        """Try to match query against templates. Returns SQL or None."""
        q = query.lower().strip()

        for name, pattern in self.PATTERNS.items():
            match = re.search(pattern["regex"], q)
            if match:
                return self._fill_template(
                    name, match, pattern["template"], schema
                )

        return None

    def _fill_template(
        self, name: str, match: re.Match, template: str, schema: DatabaseSchema
    ) -> str | None:
        """Resolve column/table names against actual schema."""
        tables = {t.name.lower(): t for t in schema.tables}
        all_columns = {}
        for t in schema.tables:
            for c in t.columns:
                all_columns[c.name.lower()] = (t.name, c.name)

        # Extract captured groups and resolve
        groups = match.groups()

        if name == "top_n_by":
            n, group_col, metric_col = groups[0], groups[1], groups[2]
            if group_col.lower() in all_columns and metric_col.lower() in all_columns:
                t1, g = all_columns[group_col.lower()]
                t2, m = all_columns[metric_col.lower()]
                if t1 == t2:
                    return template.format(
                        table=t1, group=g, metric=m, n=int(n)
                    )

        elif name == "top_n_in":
            n, group_col, table_hint = groups[0], groups[1], groups[2]
            # Try to resolve group_col as a column
            if group_col.lower() in all_columns:
                table_name, col_name = all_columns[group_col.lower()]
                # If table_hint matches the table, use it
                if table_hint.lower() in table_name.lower() or table_name.lower() in table_hint.lower():
                    return template.format(table=table_name, group=col_name, n=int(n))
                return template.format(table=table_name, group=col_name, n=int(n))
            # Try table_hint as a table name and group_col as a column in it
            for t_name, t_schema in tables.items():
                if table_hint.lower() in t_name:
                    for c in t_schema.columns:
                        if group_col.lower() in c.name.lower():
                            return template.format(table=t_schema.name, group=c.name, n=int(n))

        elif name == "top_n_simple":
            n, group_col = groups[0], groups[1]
            if group_col.lower() in all_columns:
                table_name, col_name = all_columns[group_col.lower()]
                return template.format(table=table_name, group=col_name, n=int(n))

        elif name in ("total", "average"):
            col_word = groups[-1]
            if col_word.lower() in all_columns:
                table, col = all_columns[col_word.lower()]
                return template.format(table=table, col=col)

        elif name == "count":
            entity_word = groups[-1].lower()
            # Try to match the entity word to a table name
            for t in schema.tables:
                if entity_word in t.name.lower() or t.name.lower() in entity_word:
                    return template.format(table=t.name)
            # No match found — do not fall back to an arbitrary table
            return None

        return None
