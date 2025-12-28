"""
Stage 3: Generative Reasoning Layer
Uses RAG context from stage2_rag.py
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
from typing import Dict, Any, List

from google import genai


from src.stage2_rag import (
    rag_query, 
    VECTOR_DB_DIR,
    check_rag_status,
    get_collection  
)

from src.stage1_modeling import PriceModel

# ==============================
# GEMINI CLIENT
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"


# ==============================
# PROMPT BUILDER
# ==============================
def build_investment_prompt(
    property_input: Dict[str, Any],
    prediction: Dict[str, Any],
    context_chunks: List[Dict[str, Any]],
    user_context: str,
) -> str:
    """Build structured prompt with RAG context"""
    
    # Format RAG context
    if context_chunks:
        ctx_strs = []
        for i, ch in enumerate(context_chunks[:5]):  # Top 5 chunks
            ctx_strs.append(
                f"SOURCE {i+1}: {ch['source']} (chunk {ch['chunk_id']}, relevance={1-ch['distance']:.3f})\n"
                f"{ch['text']}\n{'='*80}"
            )
        ctx_block = "\n\n".join(ctx_strs)
    else:
        ctx_block = "[NO RAG CONTEXT AVAILABLE - Provide general analysis based on property data only]"
    
    prompt = f"""You are a real estate investment analyst for Indian residential properties.

You MUST follow these rules:
- If the retrieved rag chunks does not contain evidence for a claim, write "Insufficient evidence in retrieved documents".
- Do NOT invent laws, rates, or project details.

**REGULATORY & MARKET CONTEXT (from RAG retrieval):**
{ctx_block}

**PROPERTY DETAILS:**
{json.dumps(property_input, indent=2)}

**ML MODEL PREDICTION:**
{json.dumps(prediction, indent=2)}

**INVESTOR PROFILE:**
{user_context}

**INVESTMENT QUESTION:**
"Is this property a good investment today, considering price trends, rental yield, 
legal constraints, and upcoming market developments?"

**YOUR ANALYSIS MUST ADDRESS:**
1. Rental Yield: Is this good for passive income? (3%+ = good for Indian market)
2. RERA Compliance: Any regulatory issues or requirements?
3. Floor/Construction Limits: Are there restrictions per local rules?
4. Infrastructure: Metro connectivity, roads, upcoming projects?
5. Market Trends: Price appreciation potential?
6. Legal Risks: Documentation, approvals, encumbrances?

**RATING SCALE:**
- EXCELLENT: Strong buy, multiple positive factors
- GOOD: Solid investment, minor concerns
- AVERAGE: Neutral, balanced pros/cons
- POOR: Avoid, significant red flags

**OUTPUT FORMAT (JSON only):**
{{
  "investment_view": "EXCELLENT|GOOD|AVERAGE|POOR",
  "summary": "Direct YES/NO recommendation with key reasoning (3-4 sentences)",
  "drivers": ["Key positive factor 1", "Key positive factor 2", "Key positive factor 3"],
  "risks": ["Risk factor 1", "Risk factor 2"],
  "assumptions": ["What we're assuming"],
  "rag_sources_used": ["Document names cited"]
}}
"""
    return prompt


# ==============================
# GEMINI API CALL
# ==============================
def call_gemini_for_investment(
    property_input: Dict[str, Any],
    prediction: Dict[str, Any],
    context_chunks: List[Dict[str, Any]],
    user_context: str,
) -> Dict[str, Any]:
    """Generate investment analysis using Gemini"""
    
    prompt = build_investment_prompt(
        property_input=property_input,
        prediction=prediction,
        context_chunks=context_chunks,
        user_context=user_context,
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        
        raw = response.text or ""
        
        # Clean markdown
        raw_clean = re.sub(r"^```(?:json)?", "", raw.strip())
        raw_clean = re.sub(r"```$", "", raw_clean.strip())
        
        # Parse JSON
        data = json.loads(raw_clean)
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw response: {raw[:500]}")
        return {
            "investment_view": "UNKNOWN",
            "summary": "Analysis failed - JSON parsing error.",
            "drivers": [],
            "risks": [f"Parse error: {str(e)}"],
            "assumptions": [],
            "rag_sources_used": [],
            "_raw_response": raw[:500]
        }
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {
            "investment_view": "ERROR",
            "summary": f"API call failed: {str(e)}",
            "drivers": [],
            "risks": [str(e)],
            "assumptions": [],
            "rag_sources_used": []
        }


# ==============================
# STAGE 3 PIPELINE
# ==============================
def run_stage3_pipeline(
    property_input: Dict[str, Any],
    user_context: str,
) -> Dict[str, Any]:
    """
    Complete investment analysis pipeline
    
    Returns:
        Dict with narrative, prediction, and metadata
    """
    
    print("\n" + "="*60)
    print("STAGE 3 PIPELINE: INVESTMENT ANALYSIS")
    print("="*60)
    
    # ---- STEP 1: Check RAG system ----
    print("\n[0/3] Checking RAG system...")
    print(f"Vector DB: {VECTOR_DB_DIR}")
    
    collection = get_collection()  # Get collection lazily
    print(f"Collection size: {collection.count()} chunks")
    
    if collection.count() == 0:
        print("⚠️ WARNING: RAG collection is empty!")
        print("Run: python src/stage2_rag.py")
    
    # ---- STEP 2: ML Prediction ----
    print("\n[1/3] Running ML price prediction...")
    pm = PriceModel()
    pm.train()
    
    prediction = pm.predict_for_property({
        "size_sqft": property_input["size_sqft"],
        "age_yrs": property_input["age_yrs"],
        "locality": property_input["locality"],
        "property_type": property_input["property_type"],
    })
    
    predicted_price = prediction.get('predicted_price', 0)
    print(f"✓ Predicted price: ₹{predicted_price:,.0f}")
    
    # ---- STEP 3: RAG Retrieval ----
    print("\n[2/3] Retrieving relevant documents...")
    
    # Build query
    city = property_input.get("city", "Hyderabad")
    locality = property_input.get("locality", "")
    ptype = property_input.get("property_type", "")
    
    rag_query_text = (
        f"Telangana RERA rules real estate regulation "
        f"{city} {locality} {ptype} residential property "
        f"rental yield investment risks approvals infrastructure metro "
        f"floor limits construction regulations"
    )
    
    print(f"Query: {rag_query_text[:100]}...")
    
    context_chunks = rag_query(rag_query_text, top_k=5)
    print(f"✓ Retrieved {len(context_chunks)} chunks")
    
    if context_chunks:
        print("Sources:")
        for i, chunk in enumerate(context_chunks[:3], 1):
            print(f"  {i}. {chunk['source']} (relevance: {1-chunk['distance']:.3f})")
    else:
        print("⚠️ No chunks retrieved - analysis will be generic")
    
    # ---- STEP 4: LLM Analysis ----
    print("\n[3/3] Generating investment analysis...")
    
    narrative = call_gemini_for_investment(
        property_input=property_input,
        prediction=prediction,
        context_chunks=context_chunks,
        user_context=user_context,
    )
    
    print("✓ Analysis complete")
    
    return {
        "narrative": narrative,
        "prediction": prediction,
        "rag_chunks_used": len(context_chunks),
        "property_input": property_input,
        "rag_sources": [ch['source'] for ch in context_chunks] if context_chunks else []
    }


# ==============================
# MAIN - TESTING
# ==============================
if __name__ == "__main__":
    print("="*60)
    print("STAGE 3: INVESTMENT REASONING")
    print("="*60)
    
    # Check RAG status first
    check_rag_status()
    
    # Test property
    example_property = {
        "city": "Hyderabad",
        "locality": "Madhapur",
        "size_sqft": 1200,
        "property_type": "Apartment",
        "age_yrs": 3.5
    }
    
    user_ctx = "Long-term rental investor (8+ years), moderate risk tolerance."
    
    print("\n" + "="*60)
    print("RUNNING PIPELINE")
    print("="*60)
    
    result = run_stage3_pipeline(example_property, user_ctx)
    
    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(json.dumps(result["narrative"], indent=2))
    print(f"\nMetadata:")
    print(f"  Predicted Price: ₹{result['prediction'].get('predicted_price', 0):,.0f}")
    print(f"  RAG Chunks Used: {result['rag_chunks_used']}")
    print(f"  Sources: {', '.join(result['rag_sources']) if result['rag_sources'] else 'None'}")