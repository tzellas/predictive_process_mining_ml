import pandas as pd
import pm4py
import csv


def read_clean_log(filename: str, trace_identifier: str = "case:concept:name"):

  xes_log  = pm4py.read_xes(filename)
  df_log = pm4py.convert_to_dataframe(xes_log)

  # sort traces by trace id and timestamp
  df_log = df_log.sort_values([trace_identifier, "time:timestamp"]).reset_index(drop=True)

  # keep only complete events if life cycle column is present
  if "lifecycle:transition" in df_log.columns:
    df_log = df_log[
        df_log["lifecycle:transition"].astype(str).str.lower() == "complete"
    ].reset_index(drop=True)

  # ignore trace attributes except trace id, keep event attributes
  keep_cols = [
      c for c in df_log.columns if not c.startswith("case:") or c == trace_identifier
  ]

  df_log = df_log[keep_cols].copy()
  return df_log


def build_prefixes(df_log: pd.DataFrame,  trace_identifier: str = "case:concept:name",base: int = 1, gap: int = 3):
  seen_prefixes = set()
  prefixes = []
  j_map = {}

  # iterate through events of a single trace
  for _ ,df_trace in df_log.groupby(trace_identifier, sort=False):

    if len(df_trace) <= 2:
      continue

    df_trace = df_trace.reset_index(drop=True)

    cl_list = ""
    values = {}

    # keep only indexes that match the selected bucketing
    for i in range(base, len(df_trace)-1, gap):

      if i == base:
        start = 0
      else:
        start = i-gap+1

      # process events from last to next gap
      for event_index in range(start,i+1):

        event = df_trace.iloc[event_index]
        if event_index == 0:
          cl_list = f"{event['concept:name']}"
        else:
          cl_list += f",{event["concept:name"]}"

        for key, value in event.items():
          if key in {"concept:name", trace_identifier}:
            continue
          if key not in j_map:
            j_map[key] = ''.join([part[:2] for part in key.split(':')])
          values[j_map[key]] = str(value)

      if cl_list in seen_prefixes:
        continue
      seen_prefixes.add(cl_list)

      final_prefix_string = f"{cl_list} - Values: {values} - {df_trace.iloc[i+1]["concept:name"]}"
      prefixes.append(final_prefix_string)
  return prefixes


def convert_to_csv(prefix_list: list[str], output_path: str):
  rows = []
  for row in prefix_list:
    rows.append((f"{' - '.join(row.split(' - ', 2)[:2]).strip()}",f"{row.split(' - ',2)[2].strip()}" ))

  with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "prediction"])
        w.writerows(rows)


def reduced_csv(input_path: str, output_path: str ,max_rows: int = 300):
  with open(input_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        rows.append(row)

  with open(output_path, "w", encoding="utf-8", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerows(rows)
