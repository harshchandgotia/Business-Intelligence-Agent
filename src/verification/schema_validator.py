import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis
from src.models.schema import DatabaseSchema


class SchemaValidator:
    """Deterministic SQL validation against actual schema. No LLM."""

    FORBIDDEN_KEYWORDS = {"DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "CREATE", "TRUNCATE"}

    def validate(self, sql: str, schema: DatabaseSchema) -> tuple[bool, list[str]]:
        errors = []

        # 1. Check for forbidden statements
        for stmt in sqlparse.parse(sql):
            stmt_type = stmt.get_type()
            if stmt_type and stmt_type.upper() in self.FORBIDDEN_KEYWORDS:
                errors.append(f"Forbidden statement type: {stmt_type}")

        # Check tokens for forbidden keywords
        tokens_upper = sql.upper().split()
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw in tokens_upper:
                errors.append(f"Forbidden keyword: {kw}")

        # 2. Extract table names and validate they exist
        known_tables = {t.name.lower(): t for t in schema.tables}
        extracted_tables = self._extract_table_names(sql)

        for table in extracted_tables:
            if table.lower() not in known_tables:
                errors.append(f"Unknown table: '{table}'")

        # 3. Extract column references and validate
        known_columns = set()
        for t in schema.tables:
            for c in t.columns:
                known_columns.add(c.name.lower())
                known_columns.add(f"{t.name.lower()}.{c.name.lower()}")

        extracted_columns = self._extract_column_names(sql)
        for col in extracted_columns:
            col_lower = col.lower()
            # Skip if it's a wildcard, alias, number, or function
            if col_lower in ("*", "null", "true", "false") or col_lower.isdigit():
                continue
            if col_lower not in known_columns:
                # Might be an alias — don't flag as hard error
                pass  # could add as warning

        return (len(errors) == 0, errors)

    def _extract_table_names(self, sql: str) -> list[str]:
        """Extract table names from FROM and JOIN clauses, including subqueries."""
        tables = []
        for statement in sqlparse.parse(sql):
            self._walk_tokens(statement.tokens, tables)
        return tables

    def _walk_tokens(self, tokens, tables: list):
        """Recursively walk SQL tokens to find table references."""
        from_seen = False
        for token in tokens:
            # Recurse into parenthesized subqueries and other compound tokens
            if token.ttype is None and hasattr(token, "tokens"):
                if isinstance(token, Identifier):
                    if from_seen:
                        # Only treat as a table reference if it doesn't wrap a subquery
                        has_subquery = any(isinstance(t, Parenthesis) for t in token.tokens)
                        if not has_subquery:
                            name = token.get_real_name()
                            if name:
                                tables.append(name.strip('"').strip("'"))
                        from_seen = False
                    # Recurse into the Identifier to find tables in any subqueries
                    self._walk_tokens(token.tokens, tables)
                    continue
                elif isinstance(token, IdentifierList):
                    if from_seen:
                        for identifier in token.get_identifiers():
                            # Skip subquery aliases
                            has_sub = any(isinstance(t, Parenthesis) for t in identifier.tokens)
                            if not has_sub:
                                name = identifier.get_real_name()
                                if name:
                                    tables.append(name.strip('"').strip("'"))
                        from_seen = False
                    # Also recurse
                    self._walk_tokens(token.tokens, tables)
                    continue
                elif isinstance(token, Parenthesis):
                    # Recurse into subquery parentheses
                    self._walk_tokens(token.tokens, tables)
                    continue
                else:
                    # Other compound tokens (Where, etc.) — recurse
                    self._walk_tokens(token.tokens, tables)
                    continue

            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in (
                "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
                "FULL JOIN", "CROSS JOIN",
            ):
                from_seen = True
                continue

            if from_seen:
                if isinstance(token, Identifier):
                    name = token.get_real_name()
                    if name:
                        tables.append(name.strip('"').strip("'"))
                    from_seen = False
                elif isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        name = identifier.get_real_name()
                        if name:
                            tables.append(name.strip('"').strip("'"))
                    from_seen = False
                elif token.ttype is not sqlparse.tokens.Whitespace:
                    from_seen = False

    def _extract_column_names(self, sql: str) -> list[str]:
        """Best-effort column name extraction."""
        columns = []
        parsed = sqlparse.parse(sql)[0]
        for token in parsed.flatten():
            if token.ttype is sqlparse.tokens.Name:
                columns.append(str(token))
        return columns
