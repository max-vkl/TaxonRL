import importlib

def check_for_multimodal():
    """Check if vLLM has multimodal support modules"""
    try:
        # Try to import ImageTokenizer from different possible locations
        try:
            from vllm.utils import ImageTokenizer
            print("✓ vLLM supports images via utils.ImageTokenizer")
            return True
        except (ImportError, AttributeError):
            try:
                from vllm.inputs import ImageTokenizer
                print("✓ vLLM supports images via inputs.ImageTokenizer")
                return True
            except (ImportError, AttributeError):
                try:
                    vllm_spec = importlib.util.find_spec("vllm.mm")
                    if vllm_spec is not None:
                        print("✓ vLLM has mm module, likely supports multimodal")
                        return True
                    else:
                        print("✗ vLLM doesn't have mm module")
                except (ImportError, AttributeError):
                    pass
        
        print("✗ No direct image support found in vLLM installation")
        return False
    except Exception as e:
        print(f"Error checking vLLM multimodal support: {e}")
        return False

if __name__ == "__main__":
    print(f"Checking vLLM multimodal support...")
    has_multimodal = check_for_multimodal()
    print(f"\nSummary: vLLM {'supports' if has_multimodal else 'does NOT support'} multimodal inputs")
