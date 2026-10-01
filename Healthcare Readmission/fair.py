import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from faircode.significance import significance_report, intersectional_report

# ============================================================
# HOSPITAL READMISSION BIAS AUDIT - FAIR MODEL
# Dataset: Diabetes 130-US Hospitals (1999–2008)
# https://www.kaggle.com/datasets/brandao/diabetes
#
# Protected attributes removed: race, gender, age
# Proxy variables removed:
#   payer_code          → encodes income and race via insurance
#                         type. Medicaid = low income, and
#                         Medicaid rates differ significantly
#                         by race. A clinical readmission model
#                         has no legitimate use for payment
#                         source - medical risk is independent
#                         of who is paying.
#   discharge_disposition_id → encodes post-acute care access,
#                         which is determined by insurance and
#                         geography. Minority patients are
#                         discharged home at higher rates not
#                         because they are healthier, but
#                         because SNF access is unequal.
#                         Using it penalises structural
#                         inequality as individual risk.
#   medical_specialty   → correlates with insurance type and
#                         therefore with race and income.
#                         Specialty assignment is upstream of
#                         clinical management - not an
#                         independent clinical risk signal.
#   number_inpatient    → prior hospitalisation count is
#                         partially driven by gaps in
#                         preventive care access, which are
#                         unequal by race. Retaining it would
#                         encode structural disadvantage as
#                         individual clinical history.
#
# Retained: objective clinical signals that directly measure
# this admission's severity and diabetes management -
#   time_in_hospital   → length of stay for this admission
#   num_lab_procedures → diagnostic workup intensity
#   num_procedures     → clinical procedures performed
#   num_medications    → medication burden this admission
#   number_diagnoses   → comorbidity count
#   number_emergency   → emergency visits (not inpatient, less
#                         biased as it captures acute events)
#   max_glu_serum      → glucose reading this admission
#   A1Cresult          → HbA1c result: direct diabetes control
#   insulin / change / → treatment decisions: clinical signals
#   diabetesMed          with no demographic encoding
#   diag_1/2/3         → ICD diagnosis codes for this visit
#   admission_type_id  → emergency vs elective admission
#   admission_source_id → referral vs transfer vs ER
# ============================================================

df = pd.read_csv(Path(__file__).parent / 'diabetic_data.csv')

# Remove invalid entries
df = df[~df['race'].isin(['?'])]
df = df[df['gender'] != 'Unknown/Invalid']

# Target: 1 = readmitted within 30 days (flagged as high-risk)
#         0 = not readmitted within 30 days
df['target'] = (df['readmitted'] == '<30').astype(int)

# Protected attribute flags - retained for fairness measurement only
df['is_female']     = (df['gender'] == 'Female').astype(int)
df['is_minority']   = (~df['race'].isin(['Caucasian', 'Asian'])).astype(int)
df['age_numeric']   = df['age'].str.extract(r'\[(\d+)').astype(int)
df['is_elderly']    = (df['age_numeric'] >= 70).astype(int)

# ── PROXY VARIABLE ANALYSIS (retained for comparison) ────────
print("=" * 62)
print("PROXY VARIABLE ANALYSIS - WHY EACH FEATURE WAS DROPPED")
print("=" * 62)

print("""
  race           → protected. Section 1557 of the ACA and
                   Title VI prohibit race-based distinctions
                   in any federally-funded healthcare programme.

  gender         → protected. Sex discrimination in healthcare
                   is prohibited under Section 1557 of the ACA.
                   A clinical readmission model predicting
                   medical risk has no legitimate use for gender.

  age            → protected. The Age Discrimination Act
                   prohibits age-based disparities in federally-
                   funded programmes. Age also correlates
                   directly with comorbidity count (retained),
                   so its clinical signal is captured without
                   the protected-class risk.

  payer_code     → proxy: encodes income and race via insurance
                   type. Medicaid rates: Hispanic 9.0%,
                   AfricanAmerican 5.5%, Caucasian 2.7%.
                   A readmission model should assess clinical
                   risk - not penalise patients for their
                   insurer. Payment source has no causal
                   relationship to physiological readmission
                   risk.

  discharge_disposition_id → proxy: SNF access encodes
                   insurance and geography. Caucasian patients
                   discharged to SNFs at 17.3% vs AfricanAmerican
                   at 10.7%. Lower SNF rates → higher home
                   readmission risk - but this is a resource-
                   access gap, not a clinical one. Keeping this
                   feature lets the model penalise patients for
                   lacking access to post-acute care.

  medical_specialty → proxy: encodes insurance type and
                   geography. Specialty access is upstream of
                   clinical management and is determined by
                   structural factors outside the patient's
                   control. Not a valid individual clinical
                   risk predictor.

  number_inpatient → proxy: prior hospitalisation count is
                   higher for AfricanAmerican patients (0.70
                   vs 0.48 for Asian patients). A meaningful
                   fraction of this gap reflects differential
                   access to preventive care - penalizing
                   patients for structural underinvestment in
                   their communities, not individual health
                   behaviour or severity.
""")

# Encode categoricals - only clinical ones remain
cat_cols = [
    'diag_1', 'diag_2', 'diag_3',
    'max_glu_serum', 'A1Cresult',
    'insulin', 'change', 'diabetesMed'
]
le = LabelEncoder()
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

# ── FEATURES - FAIR (clinical signals only) ──────────────────
features = [
    # race                    removed ✓ (protected attribute)
    # gender                  removed ✓ (protected attribute)
    # age                     removed ✓ (protected attribute)
    # payer_code              removed ✓ (proxy: encodes income + race)
    # discharge_disposition_id removed ✓ (proxy: encodes post-acute access)
    # medical_specialty       removed ✓ (proxy: encodes insurance/geography)
    # number_inpatient        removed ✓ (proxy: encodes preventive care access gap)
    'admission_type_id',       # retained: emergency vs elective admission
    'admission_source_id',     # retained: ER vs referral vs transfer
    'time_in_hospital',        # retained: severity proxy - days in hospital
    'num_lab_procedures',      # retained: diagnostic intensity this visit
    'num_procedures',          # retained: clinical procedures this visit
    'num_medications',         # retained: medication burden this visit
    'number_outpatient',       # retained: outpatient visits (access to care signal -
                               #           its raw group-mean gap by race is actually
                               #           larger than number_inpatient's, 0.161 vs
                               #           0.044, so retained on clinical-relevance
                               #           grounds, not because it's less stratified)
    'number_emergency',        # retained: emergency visits (acute events, not elective)
    'number_diagnoses',        # retained: comorbidity count - captures age signal
                               #           without the protected-class risk
    'max_glu_serum',           # retained: glucose reading this admission
    'A1Cresult',               # retained: HbA1c - direct diabetes control measure
    'insulin',                 # retained: treatment decision this visit
    'change',                  # retained: medication change flag this visit
    'diabetesMed',             # retained: on diabetes medication flag
    'diag_1',                  # retained: primary ICD diagnosis code
    'diag_2',                  # retained: secondary ICD diagnosis code
    'diag_3',                  # retained: tertiary ICD diagnosis code
]

X = df_enc[features]
y = df_enc['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ── MEASURE FAIRNESS GAPS ────────────────────────────────────
results = X_test.copy()
results['pred']       = model.predict(X_test)
results['is_female']  = df.loc[X_test.index, 'is_female'].values
results['is_minority'] = df.loc[X_test.index, 'is_minority'].values
results['is_elderly'] = df.loc[X_test.index, 'is_elderly'].values

sex_flag   = results.groupby('is_female')['pred'].mean()
race_flag  = results.groupby('is_minority')['pred'].mean()
age_flag   = results.groupby('is_elderly')['pred'].mean()

print("=" * 62)
print("FAIR MODEL - MITIGATED RESULTS")
print("=" * 62)
print(f"\nModel Accuracy: {accuracy:.2%}\n")

print("── High-Risk Flag Rate by Gender ────────────────────────")
print(f"  Male patients      : {sex_flag[0]:.2%} flagged high-risk")
print(f"  Female patients    : {sex_flag[1]:.2%} flagged high-risk")
sex_sig = significance_report(results[results['is_female'] == 0]['pred'],
                              results[results['is_female'] == 1]['pred'])
print(f"\n  New Fairness Gap (Gender): {sex_sig['gap']:.2%}")
print(f"  95% CI: [{sex_sig['ci_low']:.2%}, {sex_sig['ci_high']:.2%}] (bootstrap, n=10,000 resamples)")
print(f"  Permutation p-value: {sex_sig['p_value']:.4f} "
      f"({'significant' if sex_sig['significant'] else 'not significant'} at α=0.05)")
if sex_sig['small_sample_warning']:
    print(f"  Small-sample warning: n={sex_sig['n_a']} vs {sex_sig['n_b']} (<30)")

print("\n── High-Risk Flag Rate by Race ──────────────────────────")
print(f"  Caucasian/Asian    : {race_flag[0]:.2%} flagged high-risk")
print(f"  Other minorities   : {race_flag[1]:.2%} flagged high-risk")
race_sig = significance_report(results[results['is_minority'] == 0]['pred'],
                               results[results['is_minority'] == 1]['pred'])
print(f"\n  New Fairness Gap (Race): {race_sig['gap']:.2%}")
print(f"  95% CI: [{race_sig['ci_low']:.2%}, {race_sig['ci_high']:.2%}] (bootstrap, n=10,000 resamples)")
print(f"  Permutation p-value: {race_sig['p_value']:.4f} "
      f"({'significant' if race_sig['significant'] else 'not significant'} at α=0.05)")
if race_sig['small_sample_warning']:
    print(f"  Small-sample warning: n={race_sig['n_a']} vs {race_sig['n_b']} (<30)")

print("\n── High-Risk Flag Rate by Age ───────────────────────────")
print(f"  Under 70           : {age_flag[0]:.2%} flagged high-risk")
print(f"  70+ (elderly)      : {age_flag[1]:.2%} flagged high-risk")
age_sig = significance_report(results[results['is_elderly'] == 0]['pred'],
                              results[results['is_elderly'] == 1]['pred'])
print(f"\n  New Fairness Gap (Age): {age_sig['gap']:.2%}")
print(f"  95% CI: [{age_sig['ci_low']:.2%}, {age_sig['ci_high']:.2%}] (bootstrap, n=10,000 resamples)")
print(f"  Permutation p-value: {age_sig['p_value']:.4f} "
      f"({'significant' if age_sig['significant'] else 'not significant'} at α=0.05)")
if age_sig['small_sample_warning']:
    print(f"  Small-sample warning: n={age_sig['n_a']} vs {age_sig['n_b']} (<30)")

# ── INTERSECTIONAL GAP: Female × Minority ────────────────────
# Same cross as unfair.py, so the mitigation can be judged on the
# compounded gap too - proxy removal may close the marginal gaps
# without closing the intersectional one.
inter = intersectional_report(
    results['pred'],
    results['is_female'] == 1,     # disadvantaged side of gender
    results['is_minority'] == 1,   # disadvantaged side of race
)
isr = inter['intersectional']
marg_sum = abs(inter['gap_a_alone']) + abs(inter['gap_b_alone'])
print("\n── Intersectional: Female × Minority ────────────────────")
print(f"  {'Both (minority women)':<25}: {inter['cell_rates']['both']:.2%}  (n={inter['cell_sizes']['both']})")
print(f"  {'Neither (baseline)':<25}: {inter['cell_rates']['neither']:.2%}  (n={inter['cell_sizes']['neither']})")
print(f"  {'Marginal gap (sex alone)':<25}: {inter['gap_a_alone']:.2%}")
print(f"  {'Marginal gap (race alone)':<25}: {inter['gap_b_alone']:.2%}")
print(f"  {'Intersectional gap':<25}: {isr['gap']:.2%}  [CI: {isr['ci_low']:.2%}, {isr['ci_high']:.2%}]  "
      f"p={isr['p_value']:.4f} ({'significant' if isr['significant'] else 'not significant'})")
if inter['superadditive']:
    print(f"  Superadditive: yes - compounded gap exceeds the sum of marginal gaps ({marg_sum:.2%})")
else:
    print(f"  Superadditive: no - compounded gap is within the sum of marginal gaps ({marg_sum:.2%})")
if isr['small_sample_warning']:
    print(f"  Small-sample warning: doubly-disadvantaged cell n={isr['n_a']} vs baseline n={isr['n_b']} (<30)")

print("\n" + "=" * 62)
print("WHAT CHANGED")
print("=" * 62)
print("""
THE FIX: Remove protected attributes AND their structural proxies.
         Retain only clinical signals from this admission.

  race, gender, age → removed. These are protected characteristics
    under the ACA. Their presence let the model absorb demographic
    patterns in the training data as if they were independent
    clinical risk signals.

  payer_code → removed. Insurance type is a downstream effect of
    systemic inequalities in employment, immigration status, and
    wealth. A clinical model should assess physiological risk -
    not penalise patients for having Medicaid instead of a PPO.
    This is the healthcare equivalent of keeping 'relationship'
    in a benefits model: it sounds administrative, but it carries
    protected-class signal.

  discharge_disposition_id → removed. Where a patient goes after
    discharge depends on their insurance, geography, and family
    support - not their clinical trajectory. A model that learns
    'Black patients go home → higher readmission risk' is encoding
    a resource gap as a patient-level risk factor. The causal
    direction is backwards: the gap creates the risk, the patient
    does not bring the risk to the gap.

  medical_specialty → removed. Specialty assignment reflects who
    can access specialist care. That is determined by insurance
    type and zip code - not by clinical presentation alone.
    Keeping it would allow the model to score patients on their
    access to care as if it were their clinical complexity.

  number_inpatient → removed. Prior inpatient visits for
    AfricanAmerican patients average 0.70 vs 0.48 for Asian
    patients. Part of this difference reflects the consequences
    of underinvestment in preventive care in minority communities -
    a structural gap, not an individual risk attribute. Retaining
    it amplifies that gap in every future prediction.

What remains: the clinical record of this admission - diagnosis
codes, lab and procedure counts, glucose control, medication
management, and how the patient arrived. These are the signals a
clinician would use to assess readmission risk without reference
to who the patient is demographically.

Key Insight: Healthcare readmission models don't need race or
gender to discriminate by them. Payer code, discharge destination,
and prior inpatient visits are the 'occupation' and 'relationship'
of clinical AI - variables that look like neutral operational
data but encode structural inequalities in insurance, geography,
and access to preventive care.
""")