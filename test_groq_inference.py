#!/usr/bin/env python3
"""Test script for Groq inference with Llama 3.1 8B."""

import json
import os
from groq import Groq

def test_groq_inference():
    """Test basic Groq inference with Llama 3.1 8B."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        print("   Run: export GROQ_API_KEY='your-key-here'")
        return False
    
    print(f"✅ GROQ_API_KEY detected")
    
    try:
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized")
        
        # Test message
        test_prompt = """You are an ATC optimization expert. 
        Return a JSON response with exactly these keys: model, task, status.
        Example: {"model": "llama-3.1-8b", "task": "atc_optimization", "status": "ready"}"""
        
        print("\n📨 Sending test request to Groq API (llama-3.1-8b-instant)...")
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": test_prompt
                }
            ],
            temperature=0,
            max_tokens=200,
        )
        
        response_text = completion.choices[0].message.content
        print(f"\n✅ Got response from model")
        print(f"   Response: {response_text[:100]}...")
        
        # Try to parse JSON
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                print(f"\n✅ JSON parsed successfully")
                print(f"   Model response: {parsed}")
                return True
            else:
                print(f"⚠️  No JSON found in response")
                return False
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON: {e}")
            print(f"   Response text: {response_text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_atc_inference():
    """Test ATC-specific inference."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    
    try:
        client = Groq(api_key=api_key)
        
        print("\n\n🛫 Testing ATC inference prompt...")
        
        atc_prompt = """{
  "task": "delhi_monsoon_recovery_easy",
  "briefing": "Sample ATC runway schedule conflict scenario",
  "flights": 8,
  "heuristic_plan": [{"flight_id": "AI001", "runway": "09L", "assigned_minute": 5}]
}

You are optimizing an ATC runway schedule. Return strict JSON with keys: rationale, proposal.
proposal is an array of objects with flight_id, runway, assigned_minute, hold_minutes."""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative ATC planner. Output strict JSON only."
                },
                {
                    "role": "user",
                    "content": atc_prompt
                }
            ],
            temperature=0,
            max_tokens=500,
        )
        
        response_text = completion.choices[0].message.content
        print(f"✅ ATC inference succeeded")
        print(f"   Response length: {len(response_text)} chars")
        print(f"   First 150 chars: {response_text[:150]}...")
        
        # Verify JSON structure
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            has_rationale = "rationale" in parsed
            has_proposal = "proposal" in parsed
            print(f"   Has 'rationale': {has_rationale}")
            print(f"   Has 'proposal': {has_proposal}")
            
            if has_rationale and has_proposal:
                print(f"✅ ATC response structure valid")
                return True
            else:
                print(f"⚠️  Missing required fields")
                return False
        else:
            print(f"⚠️  No JSON in response")
            return False
            
    except Exception as e:
        print(f"❌ ATC inference error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("GROQ INFERENCE TEST - Llama 3.1 8B")
    print("=" * 70)
    
    success1 = test_groq_inference()
    success2 = test_atc_inference()
    
    print("\n" + "=" * 70)
    if success1 and success2:
        print("✅ ALL TESTS PASSED - Groq inference is working!")
        print("=" * 70)
    else:
        print("⚠️  Some tests failed - check configuration above")
        print("=" * 70)
