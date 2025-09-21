#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unificador de archivos CSV por nombre de archivo.

- Recorre recursivamente una carpeta raíz, agrupando archivos por nombre (p. ej. "Migración.CSV").
- Para cada nombre, concatena todos los archivos encontrados en uno solo en la carpeta de salida.
- Detecta delimitador por archivo (csv.Sniffer) y normaliza a coma en la salida.
- Escribe el encabezado una sola vez por archivo unificado.
- Intenta leer con varias codificaciones comunes (utf-8, latin-1, cp1252).
- Reporta un resumen al final y discrepancias de encabezado/tamaño de columnas.

Uso:
    python geih.py --root "c:/Users/betol/Downloads/GEIH 2024" --out-dir "unificados"

Notas:
- No requiere pandas. Usa solo la librería estándar.
- Si algunos encabezados difieren entre carpetas, el script intentará usar el primer encabezado como referencia.
  Si un archivo posterior tiene un número de columnas distinto, se reportará la discrepancia y la fila se
  adaptará (truncada o rellenada con vacío) para coincidir con el número de columnas de referencia.
"""

from __future__ import annotations

import argparse
import csv
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Tipos
Row = List[str]
Header = List[str]


def try_open_with_encodings(path: Path, encodings: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Intenta abrir un archivo probando varias codificaciones y detecta el delimitador en un muestreo.

    Nota: Esta función puede acertar con el muestreo pero fallar durante la lectura completa. Por eso se
    complementa con `read_csv_with_auto`, que reintenta al fallar.

    Retorna (encoding_usado, delimiter_detectado) o (None, None) si falla.
    """
    sample_size = 64 * 1024  # 64KB para sniffer
    for enc in encodings:
        try:
            with path.open('r', encoding=enc, newline='') as f:
                sample = f.read(sample_size)
                if not sample:
                    # archivo vacío: retornamos coma por defecto
                    return enc, ','
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                    delimiter = dialect.delimiter
                except csv.Error:
                    # fallback a coma si el sniffer no puede detectar
                    delimiter = ','
                return enc, delimiter
        except UnicodeDecodeError:
            continue
        except Exception:
            # Otros errores (permisos, etc.): continuar probando
            continue
    return None, None


def iter_rows(path: Path, encoding: str, delimiter: str) -> Tuple[Optional[Header], List[Row]]:
    """Lee el archivo CSV y retorna (header, rows).
    Si no hay header, retorna None para el header.
    """
    rows: List[Row] = []
    header: Optional[Header] = None
    with path.open('r', encoding=encoding, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first = next(reader, None)
        except csv.Error:
            # si hay error en la primera línea, no podemos continuar
            return None, []
        except UnicodeDecodeError:
            # Propagaremos este error al nivel superior para reintentar con otra codificación
            raise
        if first is None:
            return None, []
        # Detectar si la primera fila parece encabezado: heurística simple
        def looks_like_header(fields: List[str]) -> bool:
            # Si todos los campos no-numéricos, o contienen letras, suponemos que es header
            has_alpha = any(any(c.isalpha() for c in (field or '')) for field in fields)
            return has_alpha
        if looks_like_header(first):
            header = [field.strip() for field in first]
        else:
            # No parece header: tratamos la primera como fila de datos
            rows.append([(field or '').strip() for field in first])
        for row in reader:
            rows.append([(field or '').strip() for field in row])
    return header, rows


def read_csv_with_auto(path: Path, encodings: List[str]) -> Tuple[Optional[str], Optional[str], Optional[Header], List[Row]]:
    """Intenta leer un CSV probando varias codificaciones y detectando delimitador por intento.

    Retorna (encoding, delimiter, header, rows). Si no se puede leer, retorna (None, None, None, []).
    """
    sample_size = 64 * 1024
    for enc in encodings:
        try:
            with path.open('r', encoding=enc, newline='') as f:
                sample = f.read(sample_size)
                f.seek(0)
                if not sample:
                    # Archivo vacío
                    return enc, ',', None, []
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ','
            # Leer usando iter_rows (puede lanzar UnicodeDecodeError si falla durante la iteración)
            header, rows = iter_rows(path, enc, delimiter)
            return enc, delimiter, header, rows
        except UnicodeDecodeError:
            # Probar siguiente codificación
            continue
        except Exception:
            # Otras excepciones (permisos, csv mal formado en exceso, etc.)
            continue
    return None, None, None, []


def normalize_row(row: Row, target_len: int) -> Row:
    """Ajusta una fila a un tamaño objetivo, truncando o rellenando con vacío.
    """
    if len(row) == target_len:
        return row
    if len(row) > target_len:
        return row[:target_len]
    return row + [''] * (target_len - len(row))


def unify_group(
    filename: str,
    files: List[Path],
    out_dir: Path,
    encodings_try: List[str],
) -> Dict[str, object]:
    """Unifica un grupo de archivos con el mismo nombre en un solo CSV de salida.
    Devuelve estadísticas y advertencias.
    """
    out_file = out_dir / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    files_processed = 0
    warnings: List[str] = []

    canonical_header: Optional[Header] = None
    # Abrimos el archivo de salida en modo texto utf-8 y delimitador coma
    with out_file.open('w', encoding='utf-8', newline='') as fo:
        writer = csv.writer(fo, delimiter=',')
        for p in files:
            enc, delim, header, rows = read_csv_with_auto(p, encodings_try)
            if enc is None or delim is None:
                warnings.append(f"No se pudo leer (codificación) o detectar delimitador para: {p}")
                continue
            if canonical_header is None:
                canonical_header = header if header is not None else []
                # Escribir encabezado si existe alguno; si no hay encabezado en este primer archivo, no se escribirá
                if canonical_header:
                    writer.writerow(canonical_header)
            # Si el archivo actual tiene encabezado y ya teníamos uno, validar tamaños
            if header is not None and canonical_header is not None and header != canonical_header:
                if len(header) != len(canonical_header):
                    warnings.append(
                        f"Encabezado con distinto número de columnas en {p.name}: {len(header)} vs {len(canonical_header)}. Se ajustarán filas por tamaño."
                    )
                else:
                    warnings.append(
                        f"Encabezado con nombres distintos pero mismo tamaño en {p.name}. Se conservará el encabezado canónico."
                    )
            # Normalizar filas al tamaño del encabezado canónico (si existe), o al tamaño de la primera fila si no hay header
            target_len = len(canonical_header) if canonical_header is not None else (len(rows[0]) if rows else 0)
            for row in rows:
                if target_len:
                    row = normalize_row(row, target_len)
                writer.writerow(row)
                total_rows += 1
            files_processed += 1

    return {
        'archivo_unificado': str(out_file),
        'archivos_entrantes': len(files),
        'archivos_procesados': files_processed,
        'filas_totales': total_rows,
        'advertencias': warnings,
    }


def collect_files_by_name(root: Path, extensions: Tuple[str, ...]) -> Dict[str, List[Path]]:
    """Versión original (ya no usada). Conservada por compatibilidad si se necesitara.
    """
    groups: Dict[str, List[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        ignored_dirs = {'unificados', 'Unificados', '__pycache__'}
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs]
        for fname in filenames:
            if not fname.lower().endswith(tuple(ext.lower() for ext in extensions)):
                continue
            p = Path(dirpath) / fname
            groups.setdefault(fname, []).append(p)
    return groups


def strip_accents(text: str) -> str:
    """Elimina acentos/diacríticos usando NFKD.
    """
    norm = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in norm if not unicodedata.combining(c))


def simple_singular(word: str) -> str:
    """Singularización simple: elimina una 's' final si la palabra es larga (>=4) y termina en 's'.
    No intenta reglas completas de español, pero ayuda con plurales regulares (ocupados -> ocupado).
    """
    w = word
    if len(w) >= 4 and w.endswith('s'):
        return w[:-1]
    return w


def normalized_key_from_stem(stem: str) -> str:
    """Genera una clave normalizada para agrupar nombres 'parecidos'.
    - Pasa a minúsculas
    - Elimina acentos
    - Reemplaza caracteres no alfanuméricos por espacios y colapsa espacios
    - Singulariza simple cada token (quita 's' final)
    - Quita espacios al final y une sin espacios para la clave final
    """
    s = strip_accents(stem.lower())
    # Reemplazar no alfanum por espacio
    s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
    # Colapsar espacios y aplicar singular simple por token
    tokens = [t for t in s.split() if t]
    tokens = [simple_singular(t) for t in tokens]
    if not tokens:
        return ''
    # Clave sin espacios
    return ''.join(tokens)


def collect_files_by_normalized_name(root: Path, extensions: Tuple[str, ...]) -> Tuple[Dict[str, List[Path]], Dict[str, List[str]], Dict[str, str]]:
    """Recorre recursivamente y agrupa por clave normalizada del nombre (sin extensión).

    Retorna:
    - groups: clave_normalizada -> lista de rutas
    - variants: clave_normalizada -> lista de nombres de archivo originales (solo basename)
    - representative_name: clave_normalizada -> nombre de archivo elegido para salida (el más frecuente)
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    variant_counter: Dict[str, Counter] = defaultdict(Counter)
    variants: Dict[str, List[str]] = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root):
        ignored_dirs = {'unificados', 'Unificados', '__pycache__'}
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs]
        for fname in filenames:
            if not fname.lower().endswith(tuple(ext.lower() for ext in extensions)):
                continue
            stem = Path(fname).stem
            ext = Path(fname).suffix  # conserva mayúsculas/minúsculas
            key = normalized_key_from_stem(stem)
            p = Path(dirpath) / fname
            groups[key].append(p)
            variant_counter[key][fname] += 1
            variants[key].append(fname)

    representative_name: Dict[str, str] = {}
    for key, counter in variant_counter.items():
        # Elegir el nombre más frecuente; si empate, el lexicográficamente menor
        most_common = counter.most_common()
        if not most_common:
            continue
        top_count = most_common[0][1]
        candidates = sorted([name for name, c in most_common if c == top_count])
        representative_name[key] = candidates[0]

    return groups, variants, representative_name


def main():
    parser = argparse.ArgumentParser(description='Unificar CSVs por nombre de archivo desde subcarpetas.')
    parser.add_argument('--root', type=str, default=str(Path.cwd()), help='Carpeta raíz a recorrer.')
    parser.add_argument('--out-dir', type=str, default='unificados', help='Carpeta de salida donde se guardarán los unificados.')
    parser.add_argument('--ext', type=str, default='.csv;.CSV', help='Extensiones a incluir separadas por ";" (ej: .csv;.txt).')
    parser.add_argument('--encodings', type=str, default='utf-8;latin-1;cp1252', help='Lista de codificaciones a probar separadas por ";".')
    parser.add_argument('--final-merge', action='store_true', help='Si se especifica, realiza una unificación final de todos los CSV en --out-dir hacia un único CSV en --final-out.')
    parser.add_argument('--final-out', type=str, default='Unificados_final/final.csv', help='Ruta del CSV final unificado (carpeta creada si no existe).')

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    extensions = tuple([e.strip() for e in args.ext.split(';') if e.strip()])
    encodings_try = [e.strip() for e in args.encodings.split(';') if e.strip()]

    if not root.exists() or not root.is_dir():
        print(f"ERROR: La ruta raíz no existe o no es una carpeta: {root}")
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raíz: {root}")
    print(f"Salida: {out_dir}")
    print(f"Extensiones: {extensions}")
    print(f"Codificaciones a probar: {encodings_try}")

    groups, variants, representative_name = collect_files_by_normalized_name(root, extensions)
    if not groups:
        print("No se encontraron archivos que coincidan con las extensiones dadas.")
        return

    # Mostrar mapeo de variantes si hay múltiples variantes por clave
    total_groups = len(groups)
    print(f"Se encontraron {total_groups} grupos (por nombre normalizado) para unificar.")
    for key in sorted(groups.keys()):
        reps = representative_name.get(key, None)
        unique_vars = sorted(set(variants.get(key, [])))
        if len(unique_vars) > 1:
            print(f"  Variantes agrupadas -> salida '{reps}': {unique_vars}")

    summary: Dict[str, Dict[str, object]] = {}
    # Usar el nombre representativo para el archivo de salida
    for key, files in sorted(groups.items(), key=lambda kv: representative_name.get(kv[0], kv[0])):
        out_name = representative_name.get(key, Path(files[0]).name)
        files_sorted = sorted(files, key=lambda p: str(p.parent))
        print(f"Unificando '{out_name}' con {len(files_sorted)} archivos...")
        info = unify_group(out_name, files_sorted, out_dir, encodings_try)
        summary[out_name] = info
        print(f"  -> Filas escritas: {info['filas_totales']}, Archivos procesados: {info['archivos_procesados']}/{info['archivos_entrantes']}")
        if info['advertencias']:
            print(f"  Advertencias ({len(info['advertencias'])}):")
            for w in info['advertencias']:
                print(f"    - {w}")

    # Resumen final
    print("\nResumen de unificación:")
    total_files = sum(v['archivos_entrantes'] for v in summary.values())
    total_processed = sum(v['archivos_procesados'] for v in summary.values())
    total_rows = sum(v['filas_totales'] for v in summary.values())
    print(f"- Nombres de archivo unificados: {len(summary)}")
    print(f"- Archivos entrantes (suma de todos los grupos): {total_files}")
    print(f"- Archivos procesados: {total_processed}")
    print(f"- Filas totales escritas: {total_rows}")
    print(f"- Carpeta de salida: {out_dir}")

    if args.final_merge:
        # Resolver ruta final de salida y verificar existencia
        final_out_path = Path(args.final_out)
        if not final_out_path.is_absolute():
            final_out_path = (out_dir.parent / final_out_path).resolve()
        if final_out_path.exists():
            print(f"\nEl archivo final ya existe, se omite la unificación final: {final_out_path}")
        else:
            print("\nIniciando unificación final (merge de todos los CSV unificados)...")
            perform_final_merge(out_dir, args.final_out)


def detect_keys_from_header(header: List[str]) -> Tuple[List[str], str]:
    """Dada una lista de columnas, detecta las llaves disponibles y retorna (keys, level).
    level: 'person' si incluye DIRECTORIO, SECUENCIA_P y ORDEN; 'household' si solo DIRECTORIO y SECUENCIA_P; 'unknown' caso contrario.
    """
    cols = set(h.strip().upper() for h in header)
    has_dir = 'DIRECTORIO' in cols
    has_seq = 'SECUENCIA_P' in cols
    has_ord = 'ORDEN' in cols
    if has_dir and has_seq and has_ord:
        return ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'], 'person'
    if has_dir and has_seq:
        return ['DIRECTORIO', 'SECUENCIA_P'], 'household'
    if has_dir:
        return ['DIRECTORIO'], 'dwelling'
    return [], 'unknown'


def slugify_name(name: str) -> str:
    stem = Path(name).stem
    stem = strip_accents(stem)
    slug = ''.join(ch if ch.isalnum() else '_' for ch in stem)
    slug = '_'.join([t for t in slug.split('_') if t])
    return slug.upper()


def perform_final_merge(unified_dir: Path, final_out_path_str: str) -> None:
    unified_dir = unified_dir.resolve()
    final_out_path = Path(final_out_path_str)
    if not final_out_path.is_absolute():
        final_out_path = (unified_dir.parent / final_out_path).resolve()
    final_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Enumerar CSVs de unified_dir
    csv_files = [p for p in unified_dir.iterdir() if p.is_file() and p.suffix.lower() == '.csv']
    if not csv_files:
        print(f"No se encontraron CSVs en {unified_dir}")
        return

    print(f"Archivos a integrar: {len(csv_files)}")

    # Leer encabezados rápidamente y detectar llaves
    file_meta = []  # lista de (path, header, keys, level)
    for p in sorted(csv_files):
        with p.open('r', encoding='utf-8', newline='') as f:
            first_line = f.readline().strip('\n\r')
        header = [h.strip() for h in first_line.split(',')] if first_line else []
        keys, level = detect_keys_from_header(header)
        file_meta.append((p, header, keys, level))
        print(f"- {p.name}: nivel={level}, llaves={keys}")

    # Elegir base: preferir un archivo de nivel persona con mayor número de filas, si existe; sino, nivel hogar
    person_candidates = [m for m in file_meta if m[2] == ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']]
    base_path = None
    if person_candidates:
        # Estimar filas por tamaño como heurística rápida (ya que conocemos son salidas intermedias grandes)
        person_candidates_sorted = sorted(person_candidates, key=lambda m: m[0].stat().st_size, reverse=True)
        base_path = person_candidates_sorted[0][0]
    else:
        hh_candidates = [m for m in file_meta if m[2] == ['DIRECTORIO', 'SECUENCIA_P']]
        if hh_candidates:
            hh_candidates_sorted = sorted(hh_candidates, key=lambda m: m[0].stat().st_size, reverse=True)
            base_path = hh_candidates_sorted[0][0]
    if base_path is None:
        # Como último recurso, tomar el primero
        base_path = file_meta[0][0]

    print(f"Archivo base para el merge: {base_path.name}")

    # Cargar base con dtype=str para preservar codificaciones y ceros a la izquierda
    df_final = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    base_header = df_final.columns.tolist()
    base_keys, base_level = detect_keys_from_header(base_header)

    # Merge iterativo
    for p, header, keys, level in file_meta:
        if p == base_path:
            continue
        # Leer dataframe a unir
        df_other = pd.read_csv(p, dtype=str, keep_default_na=False)
        # Determinar keys de unión compatibles con las de la base
        join_keys = []
        if base_level == 'person':
            # persona base: si other tiene ORDEN, usa las 3; si no, intenta hogar
            if keys == ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']:
                join_keys = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']
            elif keys == ['DIRECTORIO', 'SECUENCIA_P']:
                join_keys = ['DIRECTORIO', 'SECUENCIA_P']
            elif 'DIRECTORIO' in df_other.columns:
                join_keys = ['DIRECTORIO']
        elif base_level == 'household':
            # hogar base: si other tiene hogar, usa hogar; si tiene persona, expande por hogar
            if keys == ['DIRECTORIO', 'SECUENCIA_P']:
                join_keys = ['DIRECTORIO', 'SECUENCIA_P']
            elif keys == ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']:
                join_keys = ['DIRECTORIO', 'SECUENCIA_P']
            elif 'DIRECTORIO' in df_other.columns:
                join_keys = ['DIRECTORIO']
        else:
            # base desconocida: intersectar columnas
            join_keys = [c for c in ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'] if c in df_final.columns and c in df_other.columns]

        if not join_keys:
            print(f"  Aviso: {p.name} no comparte llaves con la base; se omitirá para evitar un producto cartesiano.")
            continue

        # Evitar duplicar columnas clave en sufijos
        other_cols = [c for c in df_other.columns if c not in join_keys]
        # Preparar sufijo basado en nombre del archivo
        suffix = '_' + slugify_name(p.name)
        # Unir
        print(f"  Uniendo {p.name} por llaves {join_keys} ...")
        df_final = df_final.merge(df_other[join_keys + other_cols], on=join_keys, how='outer', suffixes=('', suffix))

    # Guardar CSV final
    df_final.to_csv(final_out_path, index=False)
    print(f"Archivo final guardado en: {final_out_path}")


if __name__ == '__main__':
    main()
