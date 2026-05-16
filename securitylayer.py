
import os
# from detoxify import Detoxify
import re
import json
from typing import Dict, List
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage


class EnhancedLLMGuardrails:
    """Enhanced LLM Guardrails with harmful intent detection"""
    
    def __init__(self):
        print("Initializing guardrails...")
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.0
        )
        # self.detoxify = Detoxify('original')
        print("✓ Toxicity detector loaded")
        
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b', 
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
        
        # SIMPLIFIED harmful intent keywords (easier to match)
        self.harmful_keywords = {
            'hacking': ['hack', 'crack', 'breach', 'exploit', 'break into'],
            'passwords': ['password', 'passwd', 'login', 'credential'],
            'illegal': ['bomb', 'explosive', 'weapon', 'kill', 'murder'],
            'fraud': ['steal', 'fraud', 'scam', 'phishing'],
        }
        
        self.metrics = {
            'toxic_inputs': 0,
            'toxic_outputs': 0,
            'pii_detected': 0,
            'harmful_intent_blocked': 0,
            'total_checks': 0
        }
    
    def check_harmful_intent(self, text: str) -> Dict:
        """Check for harmful intent using simple keyword matching"""
        text_lower = text.lower()
        
        # Check if text mentions hacking AND passwords together
        has_hack_keyword = any(word in text_lower for word in self.harmful_keywords['hacking'])
        has_password_keyword = any(word in text_lower for word in self.harmful_keywords['passwords'])
        
        if has_hack_keyword and has_password_keyword:
            return {
                'detected': True,
                'category': 'hacking',
                'reason': 'Harmful intent detected: attempting to hack passwords/accounts'
            }
        
        # Check for hacking keywords alone with certain phrases
        if has_hack_keyword and any(phrase in text_lower for phrase in ['how to', 'help me', 'guide me', 'teach me', 'show me']):
            return {
                'detected': True,
                'category': 'hacking',
                'reason': 'Harmful intent detected: requesting hacking guidance'
            }
        
        # Check for bomb/weapon making
        if any(word in text_lower for word in self.harmful_keywords['illegal']):
            if any(phrase in text_lower for phrase in ['how to', 'make', 'create', 'build']):
                return {
                    'detected': True,
                    'category': 'illegal_activities',
                    'reason': 'Harmful intent detected: illegal activities'
                }
        
        # Check for fraud keywords
        if any(word in text_lower for word in self.harmful_keywords['fraud']):
            if any(phrase in text_lower for phrase in ['how to', 'help me', 'guide me']):
                return {
                    'detected': True,
                    'category': 'fraud',
                    'reason': 'Harmful intent detected: fraudulent activities'
                }
        
        return {'detected': False}
    
    def check_toxicity(self, text: str) -> tuple:
        """Check if text contains toxic content using mistral"""
        try:
            prompt = f"""Analyze this text for toxic content. Reply ONLY in this exact JSON format, nothing else:
                {{"is_toxic": true/false, 
                "category": "toxicity/threat/insult/obscene/none", 
                "confidence": 0.0-1.0,
                 "reason": "brief reason"}}

                Text to analyze: "{text}"
            """

            results = self.llm.invoke([HumanMessage(content=prompt)])
            
            raw = results.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(raw)
            
            if result.get("is_toxic") and result.get("confidence", 0) > 0.7:
                return True, result.get("category", "toxicity"), result.get("confidence", 0.9)
            
            return False, None, 0.0
    
        except Exception as e:
            print(f"⚠ Toxicity check failed: {e} — defaulting to safe")
            return False, None, 0.0
    
    def detect_pii(self, text: str) -> tuple:
        """Detect and mask PII in text"""
        detected_pii = []
        masked_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detected_pii.append({
                    'type': pii_type,
                    'value': match.group(),
                    'position': match.span()
                })
                masked_text = masked_text.replace(
                    match.group(), 
                    f"[{pii_type.upper()}_REDACTED]"
                )
        
        return masked_text, detected_pii
    
    def validate_input(self, user_input: str) -> Dict:
        """Validate user input before sending to LLM"""
        self.metrics['total_checks'] += 1
        
        # Check 1: Harmful intent (FIRST!)
        intent_check = self.check_harmful_intent(user_input)
        if intent_check['detected']:
            self.metrics['harmful_intent_blocked'] += 1
            return {
                'safe': False,
                'reason': f"{intent_check['reason']} - This request could cause harm",
                'sanitized_input': None
            }
        
        # Check 2: Toxicity
        is_toxic, category, score = self.check_toxicity(user_input)
        if is_toxic:
            self.metrics['toxic_inputs'] += 1
            return {
                'safe': False,
                'reason': f'Toxic content detected: {category} (score: {score:.2f})',
                'sanitized_input': None
            }
        
        # Check 3: Detect and mask PII
        sanitized_input, pii_found = self.detect_pii(user_input)
        if pii_found:
            self.metrics['pii_detected'] += len(pii_found)
        
        return {
            'safe': True,
            'sanitized_input': sanitized_input,
            'pii_detected': pii_found
        }
    
    def validate_output(self, llm_output: str) -> Dict:
        """Validate LLM output before showing to user"""
        # Check output for harmful content
        intent_check = self.check_harmful_intent(llm_output)
        if intent_check['detected']:
            return {
                'safe': False,
                'reason': f'Output contains harmful content: {intent_check["category"]}',
                'sanitized_output': None
            }
        
        # Check toxicity
        is_toxic, category, score = self.check_toxicity(llm_output)
        if is_toxic:
            self.metrics['toxic_outputs'] += 1
            return {
                'safe': False,
                'reason': f'LLM generated toxic content: {category}',
                'sanitized_output': None
            }
        
        return {
            'safe': True,
            'sanitized_output': llm_output
        }
    
    def get_metrics(self) -> Dict:
        """Get guardrail metrics"""
        return self.metrics


class CustomGuardrails:
    """Additional custom validation rules"""
    
    def __init__(self, blocked_topics: List[str] = None):
        self.blocked_topics = [t.lower() for t in (blocked_topics or [])]
        
        # Simplified injection keywords (easier to match)
        self.injection_keywords = [
            'ignore all previous',
            'ignore previous',
            'disregard all',
            'disregard previous',
            'forget your instructions',
            'forget previous',
            'you are now',
            'new instructions',
            'system prompt',
            'reveal your prompt',
            'show your prompt',
            'bypass',
        ]
    
    def check_prompt_injection(self, text: str) -> Dict:
        """Detect potential prompt injection attempts using keyword matching"""
        text_lower = text.lower()
        
        # Check for injection keywords
        for keyword in self.injection_keywords:
            if keyword in text_lower:
                return {
                    'detected': True,
                    'keyword': keyword,
                    'reason': 'Potential prompt injection detected'
                }
        
        # Also check with regex patterns as backup
        injection_patterns = [
            r'ignore.{0,20}(previous|all|above)',
            r'disregard.{0,20}(previous|all|above)',
            r'forget.{0,20}instructions',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text_lower):
                return {
                    'detected': True,
                    'pattern': pattern,
                    'reason': 'Potential prompt injection detected'
                }
        
        return {'detected': False}