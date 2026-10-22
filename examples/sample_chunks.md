# Sample Document Chunks — Boundary Inspection

> Demonstrates chunk boundaries, token sizing, and structural preservation
> across three chunking strategies on the Property Insurance Policy Form HO-3.

## Strategy: `Fixed-Token (400 tok, 80 overlap)`
Total Chunks: **9**

### Chunk `ho3_policy_fixed_000` (400 tokens, 1878 chars)
- **Boundary Start:** `==================================================...`
- **Boundary End:** `...amily dwelling where you reside; the two, three or`
- **Metadata:** {"window_start_token": 0, "window_end_token": 400, "chunk_size_setting": 400, "overlap_setting": 80}
```text
================================================================================ HOMEOWNERS 3 – SPECIAL FORM (HO-04-00 10 24) SECTION I – PROPERTY COVERAGES & ENDORSEMENTS POLICY NUMBER: POL-HO3-2024-884192 NAMED INSURED: John and Sarah Jenkins INSURED LOCATION: 1044 Whispering Pines Way, Boulder, CO 80302 POLICY PERIOD: October 1, 2024 to October 1, 2025 ================================================================================

AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.

DEFINITIONS In this policy, "you" and "your" refer to the "named insured" shown in the Declarations and the spouse if a resident of the same household. "We", "us" and "our" refer to the Company providing this insurance. 1. "Aircraft Liability", "Hovercraft Liability", "Motor Vehicle Liability" and "Watercraft Liability", subject to certain exceptions, are not covered under Section I. 2. "Bodily injury" means bodily harm, sickness or disease, including required care, loss of services and death that results. 3. "Business" means a trade, profession or occupation engaged in on a full-time, part-time or occasional basis. 4. "Catastrophic Ground Collapse" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned. 5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property. 7. "Residence premises" means the one family dwelling where you reside; the two, three or
```

### Chunk `ho3_policy_fixed_001` (400 tokens, 2021 chars)
- **Boundary Start:** `insured structure being condemned. 5. "Occurrence"...`
- **Boundary End:** `...OVERAGE C – PERSONAL PROPERTY 1. Covered Property:`
- **Metadata:** {"window_start_token": 320, "window_end_token": 720, "chunk_size_setting": 400, "overlap_setting": 80}
```text
insured structure being condemned. 5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property. 7. "Residence premises" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

-------------------------------------------------------------------------------- COVERAGES --------------------------------------------------------------------------------

COVERAGE A – DWELLING 1. We cover: a. The dwelling on the "residence premises" shown in the Declarations, including structures attached to the dwelling; and b. Wall-to-wall carpeting, permanently installed fixtures, and heating and cooling machinery attached to the dwelling. 2. We do not cover: a. Land, including land on which the dwelling is located; b. Underground plumbing systems or water drainage pipes located outside the foundation perimeter; c. Any structure used in whole or in part for "business" purposes unless specifically endorsed.

COVERAGE B – OTHER STRUCTURES 1. We cover other structures on the "residence premises" set apart from the dwelling by clear space. This includes detached garages, guest houses, fences, retaining walls, swimming pools, and patios connected only by a fence, utility line, or similar connection. 2. We do not cover: a. Land, including land on which the other structures are located; b. Any structure rented or held for rental to any person not a tenant of the dwelling, unless used solely as a private garage; c. Any structure from which any "business" is conducted. 3. The limit of liability for this coverage will not be more than 10% of the limit of liability that applies to Coverage A.

COVERAGE C – PERSONAL PROPERTY 1. Covered Property:
```

### Chunk `ho3_policy_fixed_002` (400 tokens, 1659 chars)
- **Boundary Start:** `Any structure rented or held for rental to any per...`
- **Boundary End:** `...s. j. $1,500 on portable electronic equipment that`
- **Metadata:** {"window_start_token": 640, "window_end_token": 1040, "chunk_size_setting": 400, "overlap_setting": 80}
```text
Any structure rented or held for rental to any person not a tenant of the dwelling, unless used solely as a private garage; c. Any structure from which any "business" is conducted. 3. The limit of liability for this coverage will not be more than 10% of the limit of liability that applies to Coverage A.

COVERAGE C – PERSONAL PROPERTY 1. Covered Property: We cover personal property owned or used by an "insured" while it is anywhere in the world. After a loss and at your request, we will cover personal property owned by a guest or a residence employee while the property is in any residence occupied by an "insured". 2. Special Limits of Liability: The special limit for each category below is the total limit for each loss for all property in that category: a. $200 on money, bank notes, bullion, gold, silver, platinum, coins, medals, and scrip. b. $1,500 on securities, accounts, deeds, evidences of debt, letters of credit, notes other than bank notes, manuscripts, personal records, and passports. c. $1,500 on watercraft of all types, including their trailers, furnishings, equipment, and outboard engines. d. $1,500 on trailers or semitrailers not used with watercraft. e. $1,500 for loss by theft of jewelry, watches, furs, precious and semiprecious stones. f. $2,500 for loss by theft of firearms and related equipment. g. $2,500 for loss by theft of silverware, silver-plated ware, goldware, and pewterware. h. $2,500 on property, on the "residence premises", used primarily for "business" purposes. i. $500 on property, away from the "residence premises", used primarily for "business" purposes. j. $1,500 on portable electronic equipment that
```

## Strategy: `Paragraph-Semantic (target 400, max 550)`
Total Chunks: **11**

### Chunk `ho3_policy_para_000` (123 tokens, 592 chars)
- **Boundary Start:** `==================================================...`
- **Boundary End:** `...nce with all applicable provisions of this policy.`
- **Metadata:** {"paragraph_count": 2, "start_paragraph_idx": 0, "end_paragraph_idx": 1}
```text
================================================================================ HOMEOWNERS 3 – SPECIAL FORM (HO-04-00 10 24) SECTION I – PROPERTY COVERAGES & ENDORSEMENTS POLICY NUMBER: POL-HO3-2024-884192 NAMED INSURED: John and Sarah Jenkins INSURED LOCATION: 1044 Whispering Pines Way, Boulder, CO 80302 POLICY PERIOD: October 1, 2024 to October 1, 2025 ================================================================================

AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.
```

### Chunk `ho3_policy_para_001` (342 tokens, 1779 chars)
- **Boundary Start:** `AGREEMENT We will provide the insurance described ...`
- **Boundary End:** `...--------------------------------------------------`
- **Metadata:** {"paragraph_count": 3, "start_paragraph_idx": 1, "end_paragraph_idx": 3}
```text
AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.

DEFINITIONS In this policy, "you" and "your" refer to the "named insured" shown in the Declarations and the spouse if a resident of the same household. "We", "us" and "our" refer to the Company providing this insurance. 1. "Aircraft Liability", "Hovercraft Liability", "Motor Vehicle Liability" and "Watercraft Liability", subject to certain exceptions, are not covered under Section I. 2. "Bodily injury" means bodily harm, sickness or disease, including required care, loss of services and death that results. 3. "Business" means a trade, profession or occupation engaged in on a full-time, part-time or occasional basis. 4. "Catastrophic Ground Collapse" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned. 5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property. 7. "Residence premises" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

-------------------------------------------------------------------------------- COVERAGES --------------------------------------------------------------------------------
```

### Chunk `ho3_policy_para_002` (273 tokens, 1409 chars)
- **Boundary Start:** `--------------------------------------------------...`
- **Boundary End:** `...the limit of liability that applies to Coverage A.`
- **Metadata:** {"paragraph_count": 3, "start_paragraph_idx": 3, "end_paragraph_idx": 5}
```text
-------------------------------------------------------------------------------- COVERAGES --------------------------------------------------------------------------------

COVERAGE A – DWELLING 1. We cover: a. The dwelling on the "residence premises" shown in the Declarations, including structures attached to the dwelling; and b. Wall-to-wall carpeting, permanently installed fixtures, and heating and cooling machinery attached to the dwelling. 2. We do not cover: a. Land, including land on which the dwelling is located; b. Underground plumbing systems or water drainage pipes located outside the foundation perimeter; c. Any structure used in whole or in part for "business" purposes unless specifically endorsed.

COVERAGE B – OTHER STRUCTURES 1. We cover other structures on the "residence premises" set apart from the dwelling by clear space. This includes detached garages, guest houses, fences, retaining walls, swimming pools, and patios connected only by a fence, utility line, or similar connection. 2. We do not cover: a. Land, including land on which the other structures are located; b. Any structure rented or held for rental to any person not a tenant of the dwelling, unless used solely as a private garage; c. Any structure from which any "business" is conducted. 3. The limit of liability for this coverage will not be more than 10% of the limit of liability that applies to Coverage A.
```

## Strategy: `Hierarchical-Policy-Aware`
Total Chunks: **12**

### Chunk `ho3_policy_hier_000` (100 tokens, 464 chars)
- **Boundary Start:** `[General Policy Overview] ========================...`
- **Boundary End:** `...==================================================`
- **Metadata:** {"section_header": "General Policy Overview", "is_standalone_section": true}
```text
[General Policy Overview]
================================================================================ HOMEOWNERS 3 – SPECIAL FORM (HO-04-00 10 24) SECTION I – PROPERTY COVERAGES & ENDORSEMENTS POLICY NUMBER: POL-HO3-2024-884192 NAMED INSURED: John and Sarah Jenkins INSURED LOCATION: 1044 Whispering Pines Way, Boulder, CO 80302 POLICY PERIOD: October 1, 2024 to October 1, 2025 ================================================================================
```

### Chunk `ho3_policy_hier_001` (56 tokens, 307 chars)
- **Boundary Start:** `[AGREEMENT We will provide the insurance described...`
- **Boundary End:** `...nce with all applicable provisions of this policy.`
- **Metadata:** {"section_header": "AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.", "is_standalone_section": true}
```text
[AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.]
AGREEMENT We will provide the insurance described in this policy in return for the premium and compliance with all applicable provisions of this policy.
```

### Chunk `ho3_policy_hier_002` (628 tokens, 3078 chars)
- **Boundary Start:** `[DEFINITIONS In this policy, "you" and "your" refe...`
- **Boundary End:** `...--------------------------------------------------`
- **Metadata:** {"section_header": "DEFINITIONS In this policy, \"you\" and \"your\" refer to the \"named insured\" shown in the Declarations and the spouse if a resident of the same household. \"We\", \"us\" and \"our\" refer to the Company providing this insurance. 1. \"Aircraft Liability\", \"Hovercraft Liability\", \"Motor Vehicle Liability\" and \"Watercraft Liability\", subject to certain exceptions, are not covered under Section I. 2. \"Bodily injury\" means bodily harm, sickness or disease, including required care, loss of services and death that results. 3. \"Business\" means a trade, profession or occupation engaged in on a fulltime, parttime or occasional basis. 4. \"Catastrophic Ground Collapse\" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned. 5. \"Occurrence\" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. \"Property damage\" means physical injury to, destruction of, or loss of use of tangible property. 7. \"Residence premises\" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.", "is_standalone_section": true}
```text
[DEFINITIONS In this policy, "you" and "your" refer to the "named insured" shown in the Declarations and the spouse if a resident of the same household. "We", "us" and "our" refer to the Company providing this insurance. 1. "Aircraft Liability", "Hovercraft Liability", "Motor Vehicle Liability" and "Watercraft Liability", subject to certain exceptions, are not covered under Section I. 2. "Bodily injury" means bodily harm, sickness or disease, including required care, loss of services and death that results. 3. "Business" means a trade, profession or occupation engaged in on a fulltime, parttime or occasional basis. 4. "Catastrophic Ground Collapse" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned. 5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property. 7. "Residence premises" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.]
DEFINITIONS In this policy, "you" and "your" refer to the "named insured" shown in the Declarations and the spouse if a resident of the same household. "We", "us" and "our" refer to the Company providing this insurance. 1. "Aircraft Liability", "Hovercraft Liability", "Motor Vehicle Liability" and "Watercraft Liability", subject to certain exceptions, are not covered under Section I. 2. "Bodily injury" means bodily harm, sickness or disease, including required care, loss of services and death that results. 3. "Business" means a trade, profession or occupation engaged in on a full-time, part-time or occasional basis. 4. "Catastrophic Ground Collapse" means geological activity resulting in all of the following: the abrupt collapse of ground cover, a depression in ground cover clearly visible to the naked eye, structural damage to the covered building including the foundation, and the insured structure being condemned. 5. "Occurrence" means an accident, including continuous or repeated exposure to substantially the same general harmful conditions, which results in property damage. 6. "Property damage" means physical injury to, destruction of, or loss of use of tangible property. 7. "Residence premises" means the one family dwelling where you reside; the two, three or four family dwelling where you reside in at least one of the family units; or that part of any other building where you reside; and which is shown in the Declarations.

-------------------------------------------------------------------------------- COVERAGES --------------------------------------------------------------------------------
```
