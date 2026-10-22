# Token-Aware Chunking: Boundary Context Demonstration

> Demonstrates the critical difference between chunking with zero overlap versus
> controlled 100-token overlap on the Property Insurance Policy Form HO-3.

## 1. Boundary Context Preservation (Task 3)

### A. Without Overlap (`overlap = 0`)
```text
--- End of Chunk 0 (tokens 465 to 500) ---
" shown in the Declarations, including structures attached to the dwelling; and
   b. Wall-to-wall carpeting, permanently installed fixtures, and heating and cooling machinery attached to

--- Start of Chunk 1 (tokens 500 to 535) ---
the dwelling.
2. We do not cover:
   a. Land, including land on which the dwelling is located;
   b. Underground plumbing systems or water drainage pipes located
```

**Severed Context:** The sentence and legal condition are abruptly cut in half across the boundary.

### B. With Controlled Overlap (`overlap = 100` tokens)
```text
--- Start of Chunk 1 (starts at token 400) ---
four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

--------------------------------------------------------------------------------
COVERAGES
--------------------------------------------------------------------------------

COVERAGE
```

**Context Preserved:** Chunk 1 reproduces the entire preceding clause, allowing retrieval of Chunk 1 to answer queries without losing necessary conditions.

## 2. Example Chunks Sized by Tokens

### Chunk `doc_tok_000`
- **Token Span:** `[0 : 500]` (500 tokens)
- **Overlap with Previous:** `0` tokens
- **Overlap with Next:** `100` tokens
```text
================================================================================
HOMEOWNERS 3 – SPECIAL FORM (HO-04-00 10 24)
SECTION I – PROPERTY COVERAGES & ENDORSEMENTS
POLICY NUMBER: POL-HO3-2024-884192
NAMED INSURED: John and Sarah Jenkins
INSURED LOCATION: 1044 Whispering Pines Way, Boulder, CO 80302
POLICY PERIOD: October 1, 2024 to October 1, 2025
================================================================================

AGREEMENT
We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.

DEFINITIONS
In this policy, "you" and "your" refer to the "named insured" shown in the Declarations and the spouse if a resident of the same household. "We", "us" and "our" refer to the Company providing this insurance.
1. "Aircraft Liability", "Hovercraft Liability", "Motor Vehicle Liability" and "Watercraft Liability", subject to certain exceptions, are not covered under Section I.
2. "Bodily injury" means bodily harm, sickness or disease, including required care, loss of services and death that results.
3. "Business" means a trade, profession or occupation engaged in on a full-time, part-time or occasional basis.
4. "Catastrophic Ground Collapse" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned.
5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage.
6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property.
7. "Residence premises" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

--------------------------------------------------------------------------------
COVERAGES
--------------------------------------------------------------------------------

COVERAGE A – DWELLING
1. We cover:
   a. The dwelling on the "residence premises" shown in the Declarations, including structures attached to the dwelling; and
   b. Wall-to-wall carpeting, permanently installed fixtures, and heating and cooling machinery attached to
```

### Chunk `doc_tok_001`
- **Token Span:** `[400 : 900]` (500 tokens)
- **Overlap with Previous:** `100` tokens
- **Overlap with Next:** `100` tokens
```text
four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

--------------------------------------------------------------------------------
COVERAGES
--------------------------------------------------------------------------------

COVERAGE A – DWELLING
1. We cover:
   a. The dwelling on the "residence premises" shown in the Declarations, including structures attached to the dwelling; and
   b. Wall-to-wall carpeting, permanently installed fixtures, and heating and cooling machinery attached to the dwelling.
2. We do not cover:
   a. Land, including land on which the dwelling is located;
   b. Underground plumbing systems or water drainage pipes located outside the foundation perimeter;
   c. Any structure used in whole or in part for "business" purposes unless specifically endorsed.

COVERAGE B – OTHER STRUCTURES
1. We cover other structures on the "residence premises" set apart from the dwelling by clear space. This includes detached garages, guest houses, fences, retaining walls, swimming pools, and patios connected only by a fence, utility line, or similar connection.
2. We do not cover:
   a. Land, including land on which the other structures are located;
   b. Any structure rented or held for rental to any person not a tenant of the dwelling, unless used solely as a private garage;
   c. Any structure from which any "business" is conducted.
3. The limit of liability for this coverage will not be more than 10% of the limit of liability that applies to Coverage A.

COVERAGE C – PERSONAL PROPERTY
1. Covered Property:
   We cover personal property owned or used by an "insured" while it is anywhere in the world. After a loss and at your request, we will cover personal property owned by a guest or a residence employee while the property is in any residence occupied by an "insured".
2. Special Limits of Liability:
   The special limit for each category below is the total limit for each loss for all property in that category:
   a. $200 on money, bank notes, bullion, gold, silver, platinum, coins, medals, and scrip.
   b. $1,500 on securities, accounts, deeds, evidences of debt, letters of credit, notes other than bank notes, manuscripts, personal records, and passports.
   c. $1,500 on watercraft of all types, including their trailers, furnishings, equipment
```

### Chunk `doc_tok_002`
- **Token Span:** `[800 : 1300]` (500 tokens)
- **Overlap with Previous:** `100` tokens
- **Overlap with Next:** `100` tokens
```text
is the total limit for each loss for all property in that category:
   a. $200 on money, bank notes, bullion, gold, silver, platinum, coins, medals, and scrip.
   b. $1,500 on securities, accounts, deeds, evidences of debt, letters of credit, notes other than bank notes, manuscripts, personal records, and passports.
   c. $1,500 on watercraft of all types, including their trailers, furnishings, equipment, and outboard engines.
   d. $1,500 on trailers or semitrailers not used with watercraft.
   e. $1,500 for loss by theft of jewelry, watches, furs, precious and semiprecious stones.
   f. $2,500 for loss by theft of firearms and related equipment.
   g. $2,500 for loss by theft of silverware, silver-plated ware, goldware, and pewterware.
   h. $2,500 on property, on the "residence premises", used primarily for "business" purposes.
   i. $500 on property, away from the "residence premises", used primarily for "business" purposes.
   j. $1,500 on portable electronic equipment that reproduces, receives, or transmits audio, visual, or data signals while in or upon a motor vehicle.

COVERAGE D – LOSS OF USE
The limit of liability for Coverage D is the total limit for the coverages in 1. Additional Living Expense, 2. Fair Rental Value and 3. Civil Authority Prohibits Use below:
1. Additional Living Expense:
   If a loss covered under Section I makes that part of the "residence premises" where you reside not fit to live in, we cover any necessary increase in living expenses incurred by you so that your household can maintain its normal standard of living.
2. Fair Rental Value:
   If a loss covered under Section I makes that part of the "residence premises" rented to others or held for rental by you not fit to live in, we cover the fair rental value of such premises less any expenses that do not continue while it is not fit to live in.
3. Civil Authority Prohibits Use:
   If a civil authority prohibits you from use of the "residence premises" as a result of direct damage to neighboring premises by a Peril Insured Against, we cover the Additional Living Expense and
```
