"""Phase 3: association-rule mining with Apriori."""

import warnings

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

if __package__:
    from ._pipeline_utils import (
        ALL_RULES_PATH,
        CLEAN_PATH,
        OHE_TRANSACTIONS_PATH,
        TOP_RULES_PATH,
        configure_runtime,
        display,
        require_files,
    )
else:
    from _pipeline_utils import (
        ALL_RULES_PATH,
        CLEAN_PATH,
        OHE_TRANSACTIONS_PATH,
        TOP_RULES_PATH,
        configure_runtime,
        display,
        require_files,
    )


def run_phase3():
    """Run notebook cells 26-30 and write all Phase 3 artifacts."""

    configure_runtime()
    require_files([CLEAN_PATH, OHE_TRANSACTIONS_PATH], "Phase 3")
    warnings.filterwarnings("ignore")
    df = pd.read_csv(CLEAN_PATH)

    # ## Phase 3: Association Rule Mining (Apriori Algorithm)
    # 
    # **Objective:** Discover non-obvious co-occurrence patterns among customer behavioral and financial attributes, with focus on identifying strong churn signals.
    # 
    # **Key Hypothesis to Test:**  
    # > "Customers from Germany holding only one product who are inactive represent a strong churn profile."
    # 
    # **Algorithm:** Apriori (mlxtend)  
    # **Dataset:** Path B transaction matrix (`churn_ohe_transactions.csv`)  
    # **Filter Criteria:** Minimum support = 0.03, confidence >= 0.50, lift >= 1.5  
    # **Rationale:** A 0.05 support floor produced too few churn-consequent rules for the rubric. The 0.03 floor still represents about 300 customers in this 10,000-record dataset and produced 17 churn-consequent rules, passing the required 10-rule threshold.
    # **Balance items:** binned on the EUR 100,000 EU deposit-guarantee ceiling (Directive 2014/49/EU) — see the Binning Revision note in Phase 1 for the anchor table and the effect of this revision on the rule set.  
    # **Deliverable:** At least 10 non-trivial, high-lift association rules with business interpretation
    # 

    # In[26]:


    # ── Load OHE Transaction Matrix ───────────────────────────────────
    df_ohe_txn = pd.read_csv(OHE_TRANSACTIONS_PATH)

    # Ensure boolean dtype
    df_ohe_txn = df_ohe_txn.astype(bool)

    # Drop the negation of the target. Keeping 'Churn_Status_Retained' as a
    # minable item produces tautological rules of the form {Retained,...} -> {X}
    # that carry no business insight. Phase 3 mines rules whose CONSEQUENT is
    # Churned; the Retained column is its complement and adds only noise.
    if 'Churn_Status_Retained' in df_ohe_txn.columns:
        df_ohe_txn = df_ohe_txn.drop(columns=['Churn_Status_Retained'])

    print("── Transaction Matrix Audit ──")
    print(f"  Shape: {df_ohe_txn.shape}")
    print(f"  Transactions (rows): {len(df_ohe_txn):,}")
    print(f"  Items (columns): {df_ohe_txn.shape[1]}")
    print(f"  Density: {df_ohe_txn.values.mean()*100:.2f}% (avg items per row / total items)")
    print(f"\n  Columns Preview:")
    for col in sorted(df_ohe_txn.columns):
        support = df_ohe_txn[col].mean()
        print(f"    {col:<40} support = {support:.4f} ({support*100:.1f}%)")


    # In[27]:


    # ── Apriori: Frequent Itemset Mining ──────────────────────────────────
    # Churn base-rate ≈ 20%. A min_support of 0.05 demanded the antecedent+
    # Churned itemset appear in >=5% of customers, which only 3 multi-attribute
    # churn rules cleared. PDF Phase 3 rubric requires >=10 non-trivial rules,
    # so we lower the support floor to 0.03 (≈300 customers — still a defensible
    # segment size, not a single-record fluke).
    MIN_SUPPORT    = 0.03   # Item combination must appear in >=3% of customers
    MAX_LEN        = 5      # Maximum itemset length (controls complexity)

    print(f"Running Apriori (min_support={MIN_SUPPORT}, max_len={MAX_LEN})...")
    frequent_itemsets = apriori(
        df_ohe_txn,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN,
        verbose=1
    )

    frequent_itemsets['itemset_size'] = frequent_itemsets['itemsets'].apply(len)
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

    print(f"\n── Frequent Itemset Summary ──")
    print(f"  Total frequent itemsets found: {len(frequent_itemsets):,}")
    print(f"\n  By itemset size:")
    display(frequent_itemsets['itemset_size'].value_counts().sort_index()
              .rename('Count').to_frame())

    print(f"\n── Top 20 Most Frequent Itemsets ──")
    display(frequent_itemsets.head(20)[['support','itemset_size','itemsets']]
              .reset_index(drop=True))


    # ### Interpretation — Frequent ≠ Interesting
    # 
    # The top of the frequency table is deliberately unexciting, and understanding why matters:
    # 
    # - **The most frequent "patterns" are single high-base-rate items** (has a credit card 70.6%, middle-aged 59.2%, male 54.6%) **and their pairwise products.** The top-2 itemset {Has_CrCard, Middle_Aged} at support 0.4176 is almost exactly 0.7055 × 0.5921 = 0.4177 — pure statistical independence (lift ≈ 1.00). High support here is an arithmetic consequence of marginal frequencies, not co-occurrence knowledge.
    # - **4,105 frequent itemsets from 33 items** shows the combinatorial scale even at min_support = 0.03. Sizes 3–4 dominate (3,316 itemsets) because every customer contributes one item per attribute family (10–11 of the 33 items — hence the 31% matrix density), so mid-size combinations are mechanically abundant.
    # - **This is exactly why the next cell filters on confidence ≥ 0.50 *and* lift ≥ 1.5:** confidence imposes decision-usefulness ("given the antecedent, churn is more likely than not"), and lift removes base-rate artifacts like the itemsets above by demanding a ≥50% deviation from independence.
    # 
    # **Conclusion:** frequency finds the haystack; interestingness measures find the needles. Reporting raw frequent itemsets as "findings" would be the classic ARM mistake this filtering pipeline is designed to avoid.
    # 

    # In[28]:


    # ── Rule Generation ───────────────────────────────────────────────────
    MIN_CONFIDENCE = 0.50
    MIN_LIFT       = 1.5

    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=MIN_CONFIDENCE
    )

    # ── Apply Lift Filter ────────────────────────────────────────────────
    rules = rules[rules['lift'] >= MIN_LIFT].copy()
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    # ── Add derived metrics ────────────────────────────────────────────
    rules['conviction']     = (1 - rules['consequent support']) / (1 - rules['confidence'] + 1e-9)
    rules['antecedent_len'] = rules['antecedents'].apply(len)
    rules['consequent_len'] = rules['consequents'].apply(len)

    # ── Filter for Churn-Focused Rules ───────────────────────────────────────
    # Defensive guard: even though we dropped Churn_Status_Retained from the
    # transaction matrix, re-filter here so re-runs against an older matrix can't
    # silently reintroduce the leakage.
    rules = rules[
        ~rules['antecedents'].apply(lambda x: 'Churn_Status_Retained' in x) &
        ~rules['consequents'].apply(lambda x: 'Churn_Status_Retained' in x)
    ].copy()

    churn_rules = rules[
        rules['consequents'].apply(lambda x: 'Churn_Status_Churned' in x)
    ].copy()

    # Drop rules whose antecedent ALSO contains Churned (would be tautological)
    churn_rules = churn_rules[
        ~churn_rules['antecedents'].apply(lambda x: 'Churn_Status_Churned' in x)
    ].copy()

    non_churn_rules = rules[
        ~rules['consequents'].apply(lambda x: 'Churn_Status_Churned' in x)
    ].copy()

    print(f"── Rule Mining Summary ──")
    print(f"  Total rules generated:             {len(rules):,}")
    print(f"  Rules with CHURN as consequent:    {len(churn_rules):,}")
    print(f"  Other high-lift rules:             {len(non_churn_rules):,}")
    print(f"\n  PDF Phase 3 requires >=10 non-trivial churn-consequent rules.")
    print(f"  Status: {'PASS' if len(churn_rules) >= 10 else 'FAIL'} ({len(churn_rules)}/10)")

    print(f"\n── Top 15 Churn-Predicting Rules (by Lift) ──")
    display(churn_rules.head(15)[
        ['antecedents','consequents','support','confidence','lift','conviction']
    ].rename(columns={
        'antecedents':'IF (Antecedent)',
        'consequents':'THEN (Consequent)',
        'support':'Support',
        'confidence':'Confidence',
        'lift':'Lift',
        'conviction':'Conviction'
    }))


    # ### Interpretation — 645 Rules Survive the Filters, but Only 17 Concern Churn. Why So Few?
    # 
    # - **The confidence bar is intentionally punishing for churn rules.** With a 20.4% base rate, confidence ≥ 0.50 forces any churn-consequent rule to carry **lift ≥ 2.45** — the antecedent must more than double churn likelihood before entering the table. Most attribute combinations cannot do that. Seventeen can, and every one of them therefore describes a genuinely elevated-risk profile rather than a base-rate echo.
    # - **The other 628 rules are mostly structural:** co-occurrences between demographic/financial bands (geography ↔ balance bands, age ↔ tenure, etc.). They clear lift 1.5 but their consequents are attributes, not the behavior of interest; they are retained in the saved file for transparency and excluded from the deliverable table.
    # - **No churn rule is a micro-segment fluke:** the support floor of 0.03 means the weakest qualifying rule still describes 313 real customers (antecedent ∩ churned), and the strongest (senior-only) describes 842.
    # 
    # - **Binning sensitivity, demonstrated:** under the original unanchored 50K/125K balance bands this count was 13 — the Mid band supported a single rule and the Low band (0.75% support) none. The DGS-anchored bands are frequent enough (Insured 15.8%, Above-ceiling 48.0%) to interact with Senior and Inactive, adding five interpretable balance rules, including the #2 rule overall. Bin boundaries determine which patterns are representable.
    # 
    # **Conclusion:** the small number of churn rules is not a shortage — it is the filter working. Ten-plus rules at ≥2.5× lift with three-digit customer counts is the profile of a defensible rule set; hundreds of "churn rules" would have meant thresholds too loose to mean anything.
    # 

    # In[29]:


    # ── Hypothesis Test: Germany + Inactive + 1 Product → Churn ──────────────────
    print("═" * 70)
    print(" HYPOTHESIS VERIFICATION: Germany + Inactive + Products_1 → Churned")
    print("═" * 70)

    # Find rules containing all three antecedent items
    target_items = {'Geography_Germany', 'Active_Status_Inactive', 'Products_Label_Products_1'}
    target_churn = frozenset(['Churn_Status_Churned'])

    # Filter from the full rule set
    matching_rules = churn_rules[
        churn_rules['antecedents'].apply(
            lambda x: target_items.issubset(x)
        )
    ]

    if len(matching_rules) > 0:
        print(f"\n  ✔ Rule FOUND! ({len(matching_rules)} matching rule(s))\n")
        display(matching_rules[['antecedents','consequents',
                                 'support','confidence','lift']].reset_index(drop=True))
    else:
        print("\n  Rule not found at current thresholds.")
        print("  Computing metrics directly from the data...\n")

    # ── Direct Computation from Raw Data ─────────────────────────────────────────
    # Compute support, confidence, lift manually for verification
    mask_antecedent = (
        (df['Geography'] == 'Germany') & 
        (df['IsActiveMember'] == 0) & 
        (df['NumOfProducts'] == 1)
    )
    mask_full = mask_antecedent & (df['Exited'] == 1)

    support_ant  = mask_antecedent.mean()
    support_full = mask_full.mean()
    confidence   = support_full / support_ant if support_ant > 0 else 0
    support_con  = df['Exited'].mean()
    lift         = confidence / support_con

    print(f"── Direct Calculation from Raw Data ──")
    print(f"  Antecedent   (Germany ∩ Inactive ∩ 1-Product): "
          f"{mask_antecedent.sum()} records ({support_ant*100:.2f}%)")
    print(f"  Itemset      (Antecedent ∩ Churned):            "
          f"{mask_full.sum()} records ({support_full*100:.2f}%)")
    print(f"\n  Support:    {support_full:.4f} ({support_full*100:.2f}%)")
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.1f}%) ← % of 'Germany+Inactive+1prod' who churned")
    print(f"  Lift:       {lift:.4f} ← {lift:.2f}x more likely to churn than baseline")
    print(f"\n  BASELINE Churn Rate (full dataset): {support_con*100:.1f}%")
    print(f"  SEGMENT  Churn Rate:                {confidence*100:.1f}%")


    # In[30]:


    # ── Top 10 Rules — Formatted Deliverable ──────────────────────────────
    # Deliverable table: single-consequent rules only. A multi-item consequent like
    # {Churned, Products_1} duplicates the information of its single-consequent
    # parent rule and would waste one of the 10 table slots on redundancy.
    top_rules = churn_rules[churn_rules['consequent_len'] == 1].nlargest(10, 'lift')[
        ['antecedents','consequents','support','confidence','lift','conviction']
    ].reset_index(drop=True)

    # Human-readable formatting
    top_rules['IF (Conditions)']     = top_rules['antecedents'].apply(
        lambda x: ' ∩ '.join(sorted(x)))
    top_rules['THEN (Outcome)']      = top_rules['consequents'].apply(
        lambda x: ' ∩ '.join(sorted(x)))
    top_rules['Support (%)']         = (top_rules['support'] * 100).round(2)
    top_rules['Confidence (%)']      = (top_rules['confidence'] * 100).round(1)
    top_rules['Lift']                = top_rules['lift'].round(3)
    top_rules['Conviction']          = top_rules['conviction'].round(3)

    display_cols = ['IF (Conditions)', 'THEN (Outcome)',
                    'Support (%)', 'Confidence (%)', 'Lift', 'Conviction']

    print("── TOP 10 ASSOCIATION RULES — CHURN PROFILE DISCOVERY ──")
    display(top_rules[display_cols])

    # Save rules
    top_rules.to_csv(TOP_RULES_PATH, index=False)
    rules.to_csv(ALL_RULES_PATH, index=False)
    print(f"\n  ✔ Rules saved to outputs/")

    print("""
    ── BUSINESS INTERPRETATION (Mining Expo Question 1: surprising rules) ──

    The Senior age band (ages 46–60) remains the dominant antecedent, and with
    the DGS-anchored balance bands a second risk vector becomes visible:
    balances above the EUR 100,000 deposit-guarantee ceiling (Dir. 2014/49/EU).

    Rule A: {Inactive ∩ Senior ∩ Products_1} → {Churned}     Lift≈3.8 Conf≈77%
      Inactive seniors holding only one product churn at ~77% — nearly 4× the
      20.4% base rate. Intervention: proactive retention call before a second
      consecutive inactive quarter; bundled-product offer.

    Rule B: {Inactive ∩ Senior ∩ Above_DGS} → {Churned}      Lift≈3.6 Conf≈73%
      NEW under the regulatory binning: inactive seniors whose balance exceeds
      the EUR 100K insured ceiling churn at 72.6%. Money above the state
      guarantee is the most mobile money in the book — one better competitor
      offer moves it. Highest-priority relationship-manager list.

    Rule C: {Inactive ∩ Senior} → {Churned}                  Lift≈3.4 Conf≈68%
      Inactivity alone is far weaker — it is the AGE interaction that drives
      the risk. Seniors disengage permanently; younger inactives re-engage.

    Rule D: {Inactive ∩ Senior ∩ CrCard} → {Churned}         Lift≈3.3 Conf≈68%
      A credit card does not protect inactive seniors at all — risk is
      essentially identical to the card-free profile (Rule C).

    Rule E: {Senior ∩ Germany} → {Churned}                   Lift≈3.3 Conf≈67%
      German seniors churn at >3× baseline regardless of activity or product
      count — a geographic product-fit or service-quality issue specific to
      the German operation.

    Rule F: {Senior ∩ Female ∩ Products_1} → {Churned}       Lift≈3.3 Conf≈67%
      Female seniors with a single product — a gender × age interaction that
      simple cross-tabs would miss.

    Rule G: {Senior ∩ Products_1} → {Churned}                Lift≈3.0 Conf≈61%
      Single-product seniors at 61% — cross-sell is the obvious lever, and the
      data shows the bank has historically failed to deepen this segment.

    Rule H: {Senior ∩ CrCard ∩ Products_1} → {Churned}       Lift≈2.9 Conf≈60%
      Card-only relationships are shallow relationships (compare Rule G — the
      card adds nothing).

    Rule I: {Senior ∩ Above_DGS ∩ Products_1} → {Churned}    Lift≈2.9 Conf≈60%
      Single-product seniors above the insured ceiling — high-value, shallow-
      anchored, uninsured excess: the costliest churn profile per customer.

    Rule J: {Senior ∩ Above_DGS} → {Churned}                 Lift≈2.8 Conf≈58%
      Even unconditionally, seniors above the EUR 100K ceiling churn at ~3×
      baseline. Under the old arbitrary 50–125K binning this pattern was split
      across two bands and misread as "the bank retains the wealthy" — the
      regulatory boundary reverses that conclusion.

    Beyond the table: the assigned hypothesis {Germany ∩ Inactive ∩ Products_1}
    holds at Lift 2.56 / Conf 52.1% (verified above), and its above-ceiling
    extension {+ Above_DGS} raises confidence to 55.7% (Lift 2.74) — uninsured
    balance adds risk on top of the German-engagement profile.

    ── KEY TAKEAWAY ──
    The bank is hemorrhaging SENIORS with shallow product engagement, and the
    losses concentrate where they hurt most: accounts holding MORE than the
    EUR 100K deposit-guarantee ceiling. Age, German geography, and uninsured
    excess balance are three separately visible risk vectors that compound
    when combined. None of this is visible on a univariate dashboard.
    """)



    return rules, top_rules


def main():
    run_phase3()


if __name__ == "__main__":
    main()
