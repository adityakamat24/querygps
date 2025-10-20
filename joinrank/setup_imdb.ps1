$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "IMDb/JOB Setup for Windows" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Configuration
$DB_NAME = "imdb_job"
$DB_USER = "postgres"
$DATA_DIR = "data\imdb"

# Create directories
New-Item -ItemType Directory -Force -Path "$DATA_DIR\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA_DIR\queries" | Out-Null

Write-Host ""
Write-Host "Step 1: Downloading JOB queries..." -ForegroundColor Yellow

if (-Not (Test-Path "join-order-benchmark")) {
    Write-Host "Cloning JOB repository..."
    git clone https://github.com/gregrahn/join-order-benchmark.git
    
    # Copy queries
    Copy-Item "join-order-benchmark\*.sql" -Destination "$DATA_DIR\queries\" -Force
    Write-Host "Queries copied to $DATA_DIR\queries\" -ForegroundColor Green
} else {
    Write-Host "JOB repository already exists."
}

Write-Host ""
Write-Host "Step 2: Downloading IMDb data..." -ForegroundColor Yellow
Write-Host ""
Write-Host "IMPORTANT: You need to download the IMDb dataset manually." -ForegroundColor Yellow
Write-Host ""
Write-Host "Option 1 - Full IMDb dataset (3.6 GB):" -ForegroundColor Cyan
Write-Host "  Download from: http://homepages.cwi.nl/~boncz/job/imdb.tgz"
Write-Host "  Extract to: $DATA_DIR\raw\" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 2 - Use smaller sample (for testing):" -ForegroundColor Cyan
Write-Host "  The JOB repository includes a smaller dataset"
Write-Host "  Check: join-order-benchmark\schema.sql"
Write-Host ""

$response = Read-Host "Have you downloaded and extracted the IMDb data to $DATA_DIR\raw? (y/n)"
if ($response -ne "y") {
    Write-Host "Please download the data first, then run this script again." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Step 3: Creating PostgreSQL database..." -ForegroundColor Yellow

# Drop and create database
Write-Host "Dropping existing database (if any)..."
psql -U $DB_USER -W -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>$null

Write-Host "Creating database $DB_NAME..."
psql -U $DB_USER -W -d postgres -c "CREATE DATABASE $DB_NAME;"

Write-Host ""
Write-Host "Step 4: Creating schema..." -ForegroundColor Yellow

# JOB Schema
$schemaSQL = @"
CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name character varying,
    imdb_index character varying(3),
    name_pcode_cf character varying(11),
    name_pcode_nf character varying(11),
    surname_pcode character varying(11),
    md5sum character varying(65)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title character varying,
    imdb_index character varying(4),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note character varying(72),
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name character varying NOT NULL,
    imdb_index character varying(2),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name character varying NOT NULL,
    country_code character varying(6),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32)
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword character varying NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15)
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note character varying
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying NOT NULL,
    note character varying
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying(10) NOT NULL,
    note character varying(1)
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name character varying NOT NULL,
    imdb_index character varying(9),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying NOT NULL,
    note character varying
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title character varying NOT NULL,
    imdb_index character varying(5),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);
"@

$schemaSQL | psql -U $DB_USER -W -d $DB_NAME

Write-Host ""
Write-Host "Step 5: Loading data..." -ForegroundColor Yellow
Write-Host "This may take 10-30 minutes depending on dataset size..." -ForegroundColor Yellow

# Load data from CSV files
$tables = @(
    "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type",
    "company_name", "company_type", "complete_cast", "info_type", "keyword",
    "kind_type", "link_type", "movie_companies", "movie_info", "movie_info_idx",
    "movie_keyword", "movie_link", "name", "person_info", "role_type", "title"
)

foreach ($table in $tables) {
    $csvFile = "$DATA_DIR\raw\$table.csv"
    if (Test-Path $csvFile) {
        Write-Host "Loading $table..."
        $filePath = (Resolve-Path $csvFile).Path
        $copyCmd = "\COPY $table FROM '$filePath' DELIMITER ',' CSV HEADER;"
        $copyCmd | psql -U $DB_USER -W -d $DB_NAME
    } else {
        Write-Host "Warning: $csvFile not found, skipping..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Step 6: Creating indexes..." -ForegroundColor Yellow

$indexSQL = @"
CREATE INDEX company_id_movie_companies ON movie_companies(company_id);
CREATE INDEX company_type_id_movie_companies ON movie_companies(company_type_id);
CREATE INDEX info_type_id_movie_info_idx ON movie_info_idx(info_type_id);
CREATE INDEX info_type_id_movie_info ON movie_info(info_type_id);
CREATE INDEX info_type_id_person_info ON person_info(info_type_id);
CREATE INDEX keyword_id_movie_keyword ON movie_keyword(keyword_id);
CREATE INDEX kind_id_aka_title ON aka_title(kind_id);
CREATE INDEX kind_id_title ON title(kind_id);
CREATE INDEX linked_movie_id_movie_link ON movie_link(linked_movie_id);
CREATE INDEX link_type_id_movie_link ON movie_link(link_type_id);
CREATE INDEX movie_id_aka_title ON aka_title(movie_id);
CREATE INDEX movie_id_cast_info ON cast_info(movie_id);
CREATE INDEX movie_id_complete_cast ON complete_cast(movie_id);
CREATE INDEX movie_id_movie_companies ON movie_companies(movie_id);
CREATE INDEX movie_id_movie_info_idx ON movie_info_idx(movie_id);
CREATE INDEX movie_id_movie_keyword ON movie_keyword(movie_id);
CREATE INDEX movie_id_movie_link ON movie_link(movie_id);
CREATE INDEX movie_id_movie_info ON movie_info(movie_id);
CREATE INDEX person_id_aka_name ON aka_name(person_id);
CREATE INDEX person_id_cast_info ON cast_info(person_id);
CREATE INDEX person_id_person_info ON person_info(person_id);
CREATE INDEX person_role_id_cast_info ON cast_info(person_role_id);
CREATE INDEX role_id_cast_info ON cast_info(role_id);

VACUUM ANALYZE;
"@

$indexSQL | psql -U $DB_USER -W -d $DB_NAME

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "IMDb/JOB Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Database: $DB_NAME" -ForegroundColor Cyan
Write-Host "Queries: $DATA_DIR\queries\" -ForegroundColor Cyan
Write-Host ""