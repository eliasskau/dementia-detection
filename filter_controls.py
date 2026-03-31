"""
filter_controls.py
------------------
Identifies GOLD STANDARD healthy elderly control participants from the WLS
dataset using the 'Data - 2020' tab of WLS-data.xlsx.

WHY GOLD STANDARD ONLY?
------------------------
A participant is "gold standard" if ALL of the following are true:

  1. diagnosis == 1  ("normal cognition")
       → A panel of clinicians reviewed the full long-interview data
         (phone interview + advanced practice provider notes) at a
         consensus conference and explicitly assigned NORMAL cognition.
         This is not a screen — it is a multi-clinician adjudicated
         decision accounting for all medical conditions and symptoms.

  2. alz_consensus == 0  ("not present")
       → The same consensus panel separately ruled out Alzheimer's
         disease as a primary or contributing factor.

  3. ppv == 2  ("normal cognition via consensus")
       → The Positive Predictive Value summary confirms the consensus
         outcome. PPV=2 means the long-interview consensus panel — not
         just a TICSm screen — is the source of the normal label.
         This minimises false positives (type I error).

  4. ticsm >= 28
       → The validated telephone cognitive screen (TICSm) also passes.
         The variable key explicitly states scores below 28 "should
         undergo comprehensive assessment", so this is a double-check.

  5. stroke == 2  ("No")
       → No doctor-reported stroke history. Stroke causes acquired
         language/cognitive deficits that would confound comparison
         with PPA/dementia patients.

  6. age > 0
       → Valid age recorded (age == -2 means missing/inappropriate).

Contrast with the EXCLUDED "assumed" tier (ppv == 1):
  Those participants never completed the long interview so there is
  NO clinician consensus — they were assumed healthy purely because
  their TICSm score was above a cutoff. That is too weak a standard
  when your dementia group has full clinical diagnoses.

Outputs:
  control_ids.txt       — one participant ID per line (gold only)
  healthy_controls.csv  — full filtered dataframe with all columns
"""

import pandas as pd

EXCEL_PATH  = "WLS-data.xlsx"
SHEET_NAME  = "Data - 2020"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

print(f"Total rows loaded from '{SHEET_NAME}': {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# ── Rename columns to short handles for readability ──────────────────────────
df = df.rename(columns={
    "idtlkbnk":                                         "pid",
    "sex":                                              "sex",
    "age 2020":                                         "age",
    "Research diagnosis via consensus":                 "diagnosis",
    "MCI subtype":                                      "mci_subtype",
    "Consensus outcome for Alzheimer\u2019s disease":   "alz_consensus",
    "Research diagnosis via proxy":                     "diag_proxy",
    "Confidence of proxy diagnosis":                    "proxy_confidence",
    "Positive predictive value summary outcome":        "ppv",
    "Negative predictive value summary outcome":        "npv",
    "TICSm score":                                      "ticsm",
    "Has a doctor ever told R they had a stroke?":      "stroke",
    " Digit ordering  score  if participant never refused an item": "digit_order",
    "Version of letter fluency task":                   "fluency_version",
    "Letter fluency -  total number scored words ":     "letter_fluency_scored",
    "Letter fluency - total # words said,  scored and disqualified": "letter_fluency_total",
})

# ── Value key reference (for comments) ───────────────────────────────────────
# diagnosis:  -2=inappropriate, 1=normal, 2=MCI, 3=dementia, 4=no diagnosis
# ppv:         1=assumed normal (TICSm ok), 2=normal (consensus),
#              3=normal (proxy), 4=MCI-AD, 5=MCI non-AD,
#              6=dementia-AD, 7=dementia non-AD, 8=dem-AD proxy,
#              9=dem non-AD proxy, 11=adjusted TICSm<29 (died/refused)
# stroke:     -3=refused, -1=don't know, 1=yes, 2=no
# ticsm:      <28 → "should undergo comprehensive assessment"
# age:        -2 = inappropriate/missing

# ── Gold standard mask ────────────────────────────────────────────────────────
# See module docstring for full rationale of each criterion.
gold_mask = (
    (df["diagnosis"]    == 1)   &   # Clinician consensus: normal cognition
    (df["alz_consensus"]== 0)   &   # Alzheimer's explicitly ruled out
    (df["ppv"]          == 2)   &   # PPV confirms consensus-normal (not just TICSm)
    (df["ticsm"]        >= 28)  &   # Cognitive screen passes (double-check)
    (df["stroke"]       == 2)   &   # No stroke history
    (df["age"]          >  0)       # Valid age recorded
)

healthy_df = df[gold_mask].copy()
healthy_df["control_tier"] = "gold (consensus-confirmed)"

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  Total participants in dataset   : {len(df)}")
print(f"  Gold standard controls          : {len(healthy_df)}")
print("=" * 55)
print(f"\nAge range  : {healthy_df['age'].min():.0f} – {healthy_df['age'].max():.0f}")
print(f"Mean age   : {healthy_df['age'].mean():.1f}  ±  {healthy_df['age'].std():.1f}")
print(f"TICSm range: {healthy_df['ticsm'].min():.0f} – {healthy_df['ticsm'].max():.0f}")
print(f"Sex (1=M, 2=F): {healthy_df['sex'].value_counts().to_dict()}")
print(f"\nAll diagnosis values: {healthy_df['diagnosis'].unique()}")
print(f"All ppv values      : {healthy_df['ppv'].unique()}")
print(f"All alz_consensus   : {healthy_df['alz_consensus'].unique()}")

# ── Save ──────────────────────────────────────────────────────────────────────
control_ids = healthy_df["pid"].tolist()

with open("control_ids.txt", "w") as f:
    for pid in control_ids:
        f.write(f"{pid}\n")
print(f"\nSaved {len(control_ids)} control IDs → control_ids.txt")

healthy_df.to_csv("healthy_controls.csv", index=False)
print(f"Saved full filtered dataframe  → healthy_controls.csv")

print(f"\nFirst 10 IDs: {control_ids[:10]}")
