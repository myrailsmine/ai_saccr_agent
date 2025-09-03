#!/usr/bin/env python3
"""
Streamlit UI Test for SA-CCR Application
Tests the Streamlit application functionality through HTTP requests
"""

import requests
import time
import json
import sys

class StreamlitUITester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.session = requests.Session()
        self.tests_run = 0
        self.tests_passed = 0
        
    def run_test(self, name, test_func):
        """Run a single test"""
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                print(f"✅ Passed - {name}")
                return True
            else:
                print(f"❌ Failed - {name}")
                return False
        except Exception as e:
            print(f"❌ Failed - {name}: {str(e)}")
            return False
    
    def test_application_accessibility(self):
        """Test if the Streamlit application is accessible"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Application accessible at {self.base_url}")
                
                # Check for Streamlit-specific content
                content = response.text
                if "streamlit" in content.lower():
                    print("✅ Streamlit framework detected")
                    return True
                else:
                    print("❌ Streamlit framework not detected in response")
                    return False
            else:
                print(f"❌ Application returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Application accessibility error: {e}")
            return False
    
    def test_professional_styling(self):
        """Test for professional CSS styling elements"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for professional styling indicators
            styling_indicators = [
                "main-header",
                "gradient",
                "enterprise",
                "professional",
                "SA-CCR",
                "Risk Analytics Platform"
            ]
            
            found_indicators = []
            for indicator in styling_indicators:
                if indicator in content:
                    found_indicators.append(indicator)
            
            if len(found_indicators) >= 3:  # At least 3 indicators should be present
                print(f"✅ Professional styling elements found: {found_indicators}")
                return True
            else:
                print(f"❌ Insufficient professional styling elements: {found_indicators}")
                return False
        except Exception as e:
            print(f"❌ Professional styling test error: {e}")
            return False
    
    def test_navigation_elements(self):
        """Test for navigation elements"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for navigation elements
            nav_elements = [
                "Calculator",
                "Portfolio",
                "Optimization", 
                "Comparison",
                "Database",
                "Settings"
            ]
            
            found_nav = []
            for element in nav_elements:
                if element in content:
                    found_nav.append(element)
            
            if len(found_nav) >= 4:  # At least 4 navigation elements
                print(f"✅ Navigation elements found: {found_nav}")
                return True
            else:
                print(f"❌ Insufficient navigation elements: {found_nav}")
                return False
        except Exception as e:
            print(f"❌ Navigation elements test error: {e}")
            return False
    
    def test_ai_assistant_presence(self):
        """Test for AI assistant functionality indicators"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for AI assistant indicators
            ai_indicators = [
                "AI",
                "assistant",
                "chat",
                "query",
                "intelligent",
                "🤖"
            ]
            
            found_ai = []
            for indicator in ai_indicators:
                if indicator in content:
                    found_ai.append(indicator)
            
            if len(found_ai) >= 2:  # At least 2 AI indicators
                print(f"✅ AI assistant indicators found: {found_ai}")
                return True
            else:
                print(f"❌ Insufficient AI assistant indicators: {found_ai}")
                return False
        except Exception as e:
            print(f"❌ AI assistant presence test error: {e}")
            return False
    
    def test_saccr_content(self):
        """Test for SA-CCR specific content"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for SA-CCR specific content
            saccr_terms = [
                "SA-CCR",
                "Basel",
                "Counterparty Credit Risk",
                "Exposure at Default",
                "EAD",
                "PFE",
                "Replacement Cost",
                "Risk Weighted Assets"
            ]
            
            found_terms = []
            for term in saccr_terms:
                if term in content:
                    found_terms.append(term)
            
            if len(found_terms) >= 3:  # At least 3 SA-CCR terms
                print(f"✅ SA-CCR content found: {found_terms}")
                return True
            else:
                print(f"❌ Insufficient SA-CCR content: {found_terms}")
                return False
        except Exception as e:
            print(f"❌ SA-CCR content test error: {e}")
            return False
    
    def test_form_elements(self):
        """Test for form elements and input fields"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for form-related elements
            form_indicators = [
                "input",
                "select",
                "button",
                "form",
                "Trade ID",
                "Notional",
                "Currency",
                "Maturity"
            ]
            
            found_forms = []
            for indicator in form_indicators:
                if indicator in content:
                    found_forms.append(indicator)
            
            if len(found_forms) >= 4:  # At least 4 form indicators
                print(f"✅ Form elements found: {found_forms}")
                return True
            else:
                print(f"❌ Insufficient form elements: {found_forms}")
                return False
        except Exception as e:
            print(f"❌ Form elements test error: {e}")
            return False
    
    def test_responsive_design(self):
        """Test for responsive design elements"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for responsive design indicators
            responsive_indicators = [
                "viewport",
                "responsive",
                "mobile",
                "tablet",
                "desktop",
                "@media",
                "container",
                "grid",
                "flex"
            ]
            
            found_responsive = []
            for indicator in responsive_indicators:
                if indicator in content:
                    found_responsive.append(indicator)
            
            if len(found_responsive) >= 3:  # At least 3 responsive indicators
                print(f"✅ Responsive design elements found: {found_responsive}")
                return True
            else:
                print(f"❌ Insufficient responsive design elements: {found_responsive}")
                return False
        except Exception as e:
            print(f"❌ Responsive design test error: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test application performance metrics"""
        try:
            start_time = time.time()
            response = self.session.get(self.base_url, timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                print(f"✅ Response time: {response_time:.2f} seconds")
                
                if response_time < 5.0:  # Should load within 5 seconds
                    print("✅ Performance acceptable")
                    return True
                else:
                    print("❌ Performance too slow")
                    return False
            else:
                print(f"❌ Performance test failed - status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False

def main():
    """Main test execution"""
    print("🌐 Starting Streamlit UI Tests")
    print("=" * 50)
    
    tester = StreamlitUITester()
    
    # Run all tests
    tests = [
        ("Application Accessibility", tester.test_application_accessibility),
        ("Professional Styling", tester.test_professional_styling),
        ("Navigation Elements", tester.test_navigation_elements),
        ("AI Assistant Presence", tester.test_ai_assistant_presence),
        ("SA-CCR Content", tester.test_saccr_content),
        ("Form Elements", tester.test_form_elements),
        ("Responsive Design", tester.test_responsive_design),
        ("Performance Metrics", tester.test_performance_metrics),
    ]
    
    for test_name, test_func in tests:
        tester.run_test(test_name, test_func)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All Streamlit UI tests passed!")
        return 0
    else:
        print("❌ Some Streamlit UI tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())