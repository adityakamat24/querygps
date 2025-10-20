#!/usr/bin/env python3
"""
Generate Realistic, Complex Queries
Creates production-level SQL queries with real-world complexity
"""
import random
import json
from pathlib import Path

class RealisticQueryGenerator:
    """Generate complex, realistic SQL queries for IMDB database"""

    def __init__(self):
        # IMDB table information with realistic constraints
        self.tables = {
            'title': {
                'size': 6_500_000,  # Large table
                'columns': ['id', 'title', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code'],
                'indexes': ['id', 'kind_id', 'production_year', 'imdb_id'],
                'frequent_filters': ['kind_id', 'production_year']
            },
            'movie_info': {
                'size': 15_000_000,  # Very large fact table
                'columns': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                'indexes': ['movie_id', 'info_type_id'],
                'frequent_filters': ['info_type_id', 'info']
            },
            'cast_info': {
                'size': 36_000_000,  # Largest table
                'columns': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                'indexes': ['person_id', 'movie_id', 'role_id'],
                'frequent_filters': ['role_id', 'person_role_id']
            },
            'name': {
                'size': 4_000_000,  # Large dimension table
                'columns': ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf'],
                'indexes': ['id', 'imdb_id', 'gender'],
                'frequent_filters': ['gender', 'name']
            },
            'movie_companies': {
                'size': 2_600_000,
                'columns': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                'indexes': ['movie_id', 'company_id', 'company_type_id'],
                'frequent_filters': ['company_type_id']
            },
            'company_name': {
                'size': 235_000,
                'columns': ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf'],
                'indexes': ['id', 'country_code'],
                'frequent_filters': ['country_code', 'name']
            },
            'movie_keyword': {
                'size': 4_500_000,
                'columns': ['id', 'movie_id', 'keyword_id'],
                'indexes': ['movie_id', 'keyword_id'],
                'frequent_filters': ['keyword_id']
            },
            'keyword': {
                'size': 135_000,
                'columns': ['id', 'keyword', 'phonetic_code'],
                'indexes': ['id', 'phonetic_code'],
                'frequent_filters': ['keyword']
            },
            'movie_info_idx': {
                'size': 1_400_000,
                'columns': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                'indexes': ['movie_id', 'info_type_id'],
                'frequent_filters': ['info_type_id']
            },
            'person_info': {
                'size': 3_000_000,
                'columns': ['id', 'person_id', 'info_type_id', 'info', 'note'],
                'indexes': ['person_id', 'info_type_id'],
                'frequent_filters': ['info_type_id']
            },
            'complete_cast': {
                'size': 135_000,
                'columns': ['id', 'movie_id', 'subject_id', 'status_id'],
                'indexes': ['movie_id', 'subject_id', 'status_id'],
                'frequent_filters': ['status_id']
            },
            'aka_title': {
                'size': 360_000,
                'columns': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note'],
                'indexes': ['movie_id', 'kind_id'],
                'frequent_filters': ['kind_id', 'production_year']
            }
        }

        # Realistic join patterns based on actual IMDB usage
        self.join_patterns = [
            # Core movie information queries
            ('title', 'movie_info', 'title.id = movie_info.movie_id'),
            ('title', 'cast_info', 'title.id = cast_info.movie_id'),
            ('title', 'movie_companies', 'title.id = movie_companies.movie_id'),
            ('title', 'movie_keyword', 'title.id = movie_keyword.movie_id'),
            ('title', 'movie_info_idx', 'title.id = movie_info_idx.movie_id'),
            ('title', 'complete_cast', 'title.id = complete_cast.movie_id'),
            ('title', 'aka_title', 'title.id = aka_title.movie_id'),

            # Person-related joins
            ('cast_info', 'name', 'cast_info.person_id = name.id'),
            ('person_info', 'name', 'person_info.person_id = name.id'),

            # Company joins
            ('movie_companies', 'company_name', 'movie_companies.company_id = company_name.id'),

            # Keyword joins
            ('movie_keyword', 'keyword', 'movie_keyword.keyword_id = keyword.id'),
        ]

        # Complex filter patterns that appear in real queries
        self.filter_templates = {
            'title': [
                "title.kind_id IN (1, 2, 3)",  # Movies, TV series, etc.
                "title.production_year BETWEEN 1990 AND 2020",
                "title.production_year >= 2000",
                "title.title LIKE '%Star%'",
                "title.production_year IS NOT NULL",
            ],
            'movie_info': [
                "movie_info.info_type_id = 3",  # Genre
                "movie_info.info_type_id IN (1, 2, 3, 4)",  # Multiple info types
                "movie_info.info LIKE '%Action%'",
                "movie_info.info = 'USA'",
                "movie_info.info_type_id = 8 AND movie_info.info LIKE '%English%'",
            ],
            'cast_info': [
                "cast_info.role_id IN (1, 2)",  # Actor, actress
                "cast_info.nr_order <= 10",  # Top billing
                "cast_info.person_role_id IS NOT NULL",
                "cast_info.note IS NOT NULL",
            ],
            'name': [
                "name.gender = 'm'",
                "name.gender = 'f'",
                "name.name LIKE '%John%'",
                "name.imdb_index IS NOT NULL",
            ],
            'movie_companies': [
                "movie_companies.company_type_id = 2",  # Production companies
                "movie_companies.company_type_id IN (1, 2)",
                "movie_companies.note IS NULL",
            ],
            'company_name': [
                "company_name.country_code = '[us]'",
                "company_name.name LIKE '%Universal%'",
                "company_name.country_code IN ('[us]', '[gb]', '[de]')",
            ],
            'movie_keyword': [
                "movie_keyword.keyword_id IN (SELECT id FROM keyword WHERE keyword = 'action')",
            ],
            'keyword': [
                "keyword.keyword LIKE '%action%'",
                "keyword.keyword IN ('violence', 'murder', 'death')",
                "keyword.keyword = 'based-on-novel'",
            ],
            'movie_info_idx': [
                "movie_info_idx.info_type_id = 101",  # Ratings
                "movie_info_idx.info_type_id = 100",  # Votes
                "movie_info_idx.info::FLOAT > 7.0",
            ]
        }

    def generate_complex_query(self, num_tables: int = None, complexity_level: str = 'medium') -> dict:
        """Generate a single complex, realistic query"""

        if num_tables is None:
            if complexity_level == 'low':
                num_tables = random.randint(3, 5)
            elif complexity_level == 'medium':
                num_tables = random.randint(5, 8)
            else:  # high
                num_tables = random.randint(8, 12)

        # Start with title table (most queries involve movies)
        selected_tables = ['title']
        joins = []
        filters = []

        # Add tables based on realistic join patterns
        available_patterns = [p for p in self.join_patterns if p[0] in selected_tables or p[1] in selected_tables]

        while len(selected_tables) < num_tables and available_patterns:
            pattern = random.choice(available_patterns)
            table1, table2, join_condition = pattern

            if table1 in selected_tables and table2 not in selected_tables:
                selected_tables.append(table2)
                joins.append(join_condition)
            elif table2 in selected_tables and table1 not in selected_tables:
                selected_tables.append(table1)
                joins.append(join_condition)
            elif table1 not in selected_tables and table2 not in selected_tables:
                # Add both tables if neither is selected
                selected_tables.extend([table1, table2])
                joins.append(join_condition)

            # Update available patterns
            available_patterns = [p for p in self.join_patterns
                                if (p[0] in selected_tables and p[1] not in selected_tables) or
                                   (p[1] in selected_tables and p[0] not in selected_tables) or
                                   (p[0] not in selected_tables and p[1] not in selected_tables)]

        # Trim to exact number if we added too many
        selected_tables = selected_tables[:num_tables]

        # Add realistic filters
        num_filters = random.randint(3, min(8, len(selected_tables) * 2))
        for _ in range(num_filters):
            table = random.choice(selected_tables)
            if table in self.filter_templates:
                filter_condition = random.choice(self.filter_templates[table])
                filters.append(filter_condition)

        # Generate complex SELECT clause with aggregations and subqueries
        select_clauses = self._generate_select_clause(selected_tables, complexity_level)

        # Add ORDER BY and LIMIT for realism
        order_by = self._generate_order_by(selected_tables)
        limit_clause = f"LIMIT {random.randint(10, 1000)}" if random.random() < 0.3 else ""

        # Construct the full query
        from_clause = ", ".join(selected_tables)
        where_clause = " AND ".join(joins + filters) if joins or filters else "1=1"

        query = f"""SELECT {select_clauses}
FROM {from_clause}
WHERE {where_clause}"""

        if order_by:
            query += f"\n{order_by}"
        if limit_clause:
            query += f"\n{limit_clause}"

        # Calculate estimated cost based on table sizes and joins
        estimated_cost = self._estimate_query_cost(selected_tables, len(joins), len(filters))

        return {
            'sql': query,
            'tables': selected_tables,
            'num_joins': len(joins),
            'num_filters': len(filters),
            'complexity_level': complexity_level,
            'estimated_cost': estimated_cost,
            'estimated_rows': self._estimate_result_size(selected_tables, len(filters))
        }

    def _generate_select_clause(self, tables: list, complexity_level: str) -> str:
        """Generate realistic SELECT clause"""

        if complexity_level == 'low':
            # Simple projections
            if 'title' in tables:
                return "title.title, title.production_year"
            return f"{tables[0]}.id, {tables[0]}.*"

        elif complexity_level == 'medium':
            # Mix of projections and simple aggregations
            clauses = []
            if 'title' in tables:
                clauses.append("title.title")
            if 'name' in tables:
                clauses.append("name.name")
            if 'movie_info' in tables:
                clauses.append("COUNT(movie_info.id) as info_count")
            if len(clauses) < 2:
                clauses.extend([f"{tables[0]}.id", f"{tables[1]}.id" if len(tables) > 1 else f"{tables[0]}.*"])
            return ", ".join(clauses)

        else:  # high complexity
            # Complex aggregations and window functions
            clauses = []
            if 'title' in tables:
                clauses.append("title.title")
            if 'cast_info' in tables:
                clauses.append("COUNT(DISTINCT cast_info.person_id) as cast_count")
            if 'movie_info' in tables:
                clauses.append("STRING_AGG(movie_info.info, ', ') as genres")
            if 'company_name' in tables:
                clauses.append("company_name.name as production_company")

            # Add window function for realism
            if len(tables) >= 3:
                clauses.append("ROW_NUMBER() OVER (ORDER BY title.production_year DESC) as rank")

            if not clauses:
                clauses = [f"{tables[0]}.id", "COUNT(*) as total_count"]

            return ", ".join(clauses)

    def _generate_order_by(self, tables: list) -> str:
        """Generate realistic ORDER BY clause"""
        if random.random() < 0.7:  # 70% chance of ORDER BY
            if 'title' in tables and random.random() < 0.5:
                return "ORDER BY title.production_year DESC"
            elif 'name' in tables:
                return "ORDER BY name.name"
            else:
                return f"ORDER BY {tables[0]}.id"
        return ""

    def _estimate_query_cost(self, tables: list, num_joins: int, num_filters: int) -> float:
        """Estimate query execution cost based on table sizes and complexity"""
        base_cost = sum(self.tables[table]['size'] for table in tables if table in self.tables)
        join_cost = base_cost * (num_joins ** 1.5) * 0.00001
        filter_cost = base_cost * num_filters * 0.000001
        return base_cost * 0.000001 + join_cost + filter_cost

    def _estimate_result_size(self, tables: list, num_filters: int) -> int:
        """Estimate result set size"""
        if not tables:
            return 0
        min_table_size = min(self.tables[table]['size'] for table in tables if table in self.tables)
        # More filters = smaller result set
        reduction_factor = 0.5 ** num_filters
        return max(1, int(min_table_size * reduction_factor))

    def generate_query_set(self, num_queries: int = 100, complexity_mix: bool = True) -> list:
        """Generate a set of realistic queries with varying complexity"""
        queries = []

        if complexity_mix:
            # Mix of complexity levels for realistic workload
            complexities = (['low'] * (num_queries // 4) +
                          ['medium'] * (num_queries // 2) +
                          ['high'] * (num_queries // 4))
        else:
            complexities = ['medium'] * num_queries

        for i in range(num_queries):
            complexity = complexities[i % len(complexities)]
            query_info = self.generate_complex_query(complexity_level=complexity)
            query_info['query_id'] = f"realistic_{i+1:03d}"
            queries.append(query_info)

        return queries

def generate_realistic_dataset():
    """Generate realistic query dataset for enhanced experiments"""
    print("🔧 Generating realistic, complex query dataset...")

    generator = RealisticQueryGenerator()

    # Generate different complexity levels
    datasets = {
        'complex_low': generator.generate_query_set(50, complexity_mix=False),
        'complex_medium': generator.generate_query_set(75, complexity_mix=False),
        'complex_high': generator.generate_query_set(50, complexity_mix=False),
        'mixed_workload': generator.generate_query_set(100, complexity_mix=True)
    }

    # Create output directory
    output_dir = Path('data/imdb/realistic_queries')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    for dataset_name, queries in datasets.items():
        # Save as JSON
        with open(output_dir / f'{dataset_name}.json', 'w') as f:
            json.dump(queries, f, indent=2)

        # Save individual SQL files for each query
        sql_dir = output_dir / dataset_name
        sql_dir.mkdir(exist_ok=True)

        for query in queries:
            sql_file = sql_dir / f"{query['query_id']}.sql"
            with open(sql_file, 'w') as f:
                f.write(query['sql'])

    print(f"✅ Generated realistic query datasets:")
    for dataset_name, queries in datasets.items():
        avg_tables = sum(len(q['tables']) for q in queries) / len(queries)
        avg_joins = sum(q['num_joins'] for q in queries) / len(queries)
        print(f"   📊 {dataset_name}: {len(queries)} queries, avg {avg_tables:.1f} tables, {avg_joins:.1f} joins")

    # Generate statistics
    stats = {
        'total_queries': sum(len(queries) for queries in datasets.values()),
        'complexity_distribution': {
            level: sum(1 for dataset in datasets.values()
                      for query in dataset
                      if query['complexity_level'] == level)
            for level in ['low', 'medium', 'high']
        },
        'table_usage': {},
        'average_complexity': {}
    }

    # Calculate table usage statistics
    all_queries = [q for dataset in datasets.values() for q in dataset]
    for query in all_queries:
        for table in query['tables']:
            stats['table_usage'][table] = stats['table_usage'].get(table, 0) + 1

    # Calculate average complexity metrics
    stats['average_complexity'] = {
        'tables_per_query': sum(len(q['tables']) for q in all_queries) / len(all_queries),
        'joins_per_query': sum(q['num_joins'] for q in all_queries) / len(all_queries),
        'filters_per_query': sum(q['num_filters'] for q in all_queries) / len(all_queries),
        'estimated_cost': sum(q['estimated_cost'] for q in all_queries) / len(all_queries)
    }

    # Save statistics
    with open(output_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"📈 Dataset Statistics:")
    print(f"   • Total queries: {stats['total_queries']}")
    print(f"   • Avg tables per query: {stats['average_complexity']['tables_per_query']:.1f}")
    print(f"   • Avg joins per query: {stats['average_complexity']['joins_per_query']:.1f}")
    print(f"   • Most used tables: {sorted(stats['table_usage'].items(), key=lambda x: x[1], reverse=True)[:5]}")

    return output_dir

if __name__ == "__main__":
    generate_realistic_dataset()