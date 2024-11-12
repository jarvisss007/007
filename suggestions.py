# suggestions.py

def generate_suggestions(weekly_iv_data):
    """
    Generate suggestions based on implied volatility (IV) data for the next two months.

    Parameters:
    - weekly_iv_data (dict): Dictionary containing implied volatility data for different expirations.

    Returns:
    - suggestions (str): A string containing suggestions based on IV for the next two months.
    """
    suggestions = ""

    for exp, iv_data in weekly_iv_data.items():
        avg_iv = iv_data.get('avg_iv', None)

        if avg_iv is not None:
            if avg_iv > 0.3:
                suggestions += f"For expiry {exp}: IV is high ({avg_iv:.2f}). Consider selling strategies:\n"
                suggestions += "- Iron Condors, Covered Calls, or Strangles.\n"
            elif avg_iv < 0.2:
                suggestions += f"For expiry {exp}: IV is low ({avg_iv:.2f}). Consider buying strategies:\n"
                suggestions += "- Long Calls, Long Puts, or Straddles.\n"

    return suggestions if suggestions else "No suggestions available based on implied volatility data."

