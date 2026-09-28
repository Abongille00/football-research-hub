from dataset34_pipeline import build_all

if __name__ == "__main__":
    result = build_all(auto_download=True)
    print(result["checks"].to_string(index=False))
    print(f"Wyscout events: {result['d3_rows']:,}")
    print(f"Impect events: {result['d4_rows']:,}")
    if result.get('d3_error'): print('Wyscout loader warning:', result['d3_error'])
    if result.get('d4_error'): print('Impect loader warning:', result['d4_error'])
    print("Generated:")
    for p in result['paths']: print(" -", p)
