import asyncio
from dotenv import load_dotenv
load_dotenv()  # This forces Python to read your .env file
# Import Role 1 (Retrieval)
from src.role1_ingestion.retriever import retrieve
# Import Role 2 (Math Filters)
from src.role2_filters.filters import apply_math_filters
# Import Role 3 (Your Sandbox & Authority)
from src.role3_sandbox import apply_sandbox_filters

async def test_pipeline():
    # We use the exact query Role 2 optimized their poison against
    query = "How long do I have to return a product?"
    print(f"--- QUERY ---\n{query}\n")

    # ==========================================
    # ROLE 1: RETRIEVAL
    # ==========================================
    print("--- 1. RETRIEVAL (ROLE 1) ---")
    retrieved_chunks = retrieve(query, top_k=5)
    print(f"Retrieved {len(retrieved_chunks)} chunks.")
    for c in retrieved_chunks:
        poison_flag = "🚨 POISON" if c.get("poisoned") else "✅ CLEAN"
        print(f"  [{c['chunk_id']}] {poison_flag} | Score: {c['score']:.3f}")
    
    if not retrieved_chunks:
        print("No chunks retrieved. Databases might be empty.")
        return

    # ==========================================
    # ROLE 2: MATH FILTERS
    # ==========================================
    print("\n--- 2. MATH FILTERS (ROLE 2) ---")
    math_survivors = apply_math_filters(retrieved_chunks)
    print(f"{len(math_survivors)}/{len(retrieved_chunks)} survived Role 2.")
    for c in math_survivors:
        poison_flag = "🚨 POISON" if c.get("poisoned") else "✅ CLEAN"
        print(f"  [{c['chunk_id']}] {poison_flag}")

    if not math_survivors:
        print("Math filters dropped everything. Pipeline halted.")
        return

    # ==========================================
    # ROLE 3: SANDBOX & AUTHORITY (YOURS)
    # ==========================================
    print("\n--- 3. SANDBOX & AUTHORITY (ROLE 3) ---")
    final_survivors = await apply_sandbox_filters(math_survivors)
    print(f"{len(final_survivors)}/{len(math_survivors)} survived Role 3.")
    
    for c in final_survivors:
        poison_flag = "🚨 POISON" if c.get("poisoned") else "✅ CLEAN"
        print(f"  [{c['chunk_id']}] {poison_flag} | Auth Score: {c['authority_score']:.3f}")
        print(f"      Text: {c['text'][:70]}...")

if __name__ == "__main__":
    asyncio.run(test_pipeline())