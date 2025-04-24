import pandas as pd
import re
from pathlib import Path
import os

class ContractHelper:
    """Contract processing utility class"""
    
    @staticmethod
    def parse_contract(contract):
        """Parse contract code to (symbol, year, month)"""
        match = re.match(r"^([A-Za-z]+)(\d{2})(\d{2})$", contract)
        if not match:
            raise ValueError(f"Invalid contract format: {contract}")
        return match.group(1).upper(), int(match.group(2)), int(match.group(3))
    
    @staticmethod
    def format_contract(symbol, year, month):
        """Format contract code"""
        return f"{symbol}{year%100:02d}{month:02d}"
    
    @staticmethod
    def get_adjacent_month(year, month, offset):
        """Calculate adjacent month (handling year change)"""
        total_months = year * 12 + month - 1 + offset
        new_year = total_months // 12
        new_month = total_months % 12 + 1
        return new_year, new_month
    
    @staticmethod
    def get_next_main(year, month, main_months):
        """Get next main contract month"""
        sorted_months = sorted(main_months)
        
        current_idx = -1
        for i, m in enumerate(sorted_months):
            if m == month:
                current_idx = i
                break
        
        if current_idx == -1:
            raise ValueError(f"Month {month} not in main months list")
        
        next_idx = (current_idx + 1) % len(sorted_months)
        next_month = sorted_months[next_idx]
        
        next_year = year + (1 if next_month < month else 0)
        
        return next_year, next_month

def generate_all_spreads(existing_contracts, main_months, output_file):
    """Generate all historical spread combinations and save"""
    helper = ContractHelper()
    all_spreads = []
    
    # Identify all main contracts
    main_contracts = []
    for contract in existing_contracts:
        try:
            _, _, month = helper.parse_contract(contract)
            if month in main_months:
                main_contracts.append(contract)
        except ValueError:
            continue
    
    print(f"Identified {len(main_contracts)} main contracts")
    
    # Generate spreads for each main contract
    for main_contract in main_contracts:
        symbol, year, month = helper.parse_contract(main_contract)
        
        # 1. Calculate sub-main and sub-sub-main (quarterly cycle)
        next_main_year, next_main_month = helper.get_next_main(year, month, main_months)
        next_main = helper.format_contract(symbol, next_main_year, next_main_month)
        
        next_next_main_year, next_next_main_month = helper.get_next_main(
            next_main_year, next_main_month, main_months)
        next_next_main = helper.format_contract(symbol, next_next_main_year, next_next_main_month)
        
        # 2. Calculate natural month adjacent contracts
        prev1_year, prev1_month = helper.get_adjacent_month(year, month, -1)
        prev2_year, prev2_month = helper.get_adjacent_month(year, month, -2)
        next1_year, next1_month = helper.get_adjacent_month(year, month, 1)
        next2_year, next2_month = helper.get_adjacent_month(year, month, 2)
        
        prev1 = helper.format_contract(symbol, prev1_year, prev1_month)
        prev2 = helper.format_contract(symbol, prev2_year, prev2_month)
        next1 = helper.format_contract(symbol, next1_year, next1_month)
        next2 = helper.format_contract(symbol, next2_year, next2_month)
        
        # 3. Calculate sub-main previous month
        sub_main_prev_year, sub_main_prev_month = helper.get_adjacent_month(
            next_main_year, next_main_month, -1)
        sub_main_prev = helper.format_contract(symbol, sub_main_prev_year, sub_main_prev_month)
        
        # 4. Define 7 spread combinations
        spread_definitions = [
            ('Main-SubMain', main_contract, next_main),
            ('Main-SubSubMain', main_contract, next_next_main),
            ('PrevMonth-Main', prev1, main_contract),
            ('Main-NextMonth', main_contract, next1),
            ('Main-NextNextMonth', main_contract, next2),
            ('Main-SubMainPrevMonth', main_contract, sub_main_prev),
            ('PrevPrevMonth-Main', prev2, main_contract)
        ]
        
        # 5. Filter valid spreads
        for spread_type, contract_a, contract_b in spread_definitions:
            if contract_a in existing_contracts and contract_b in existing_contracts:
                all_spreads.append({
                    'spread_type': spread_type,
                    'main_contract': main_contract,
                    'contract_a': contract_a,
                    'contract_b': contract_b,
                    'spread_code': f"{contract_a}-{contract_b}"
                })
    
    # 6. Remove duplicates and save
    df = pd.DataFrame(all_spreads)
    df = df.drop_duplicates(subset=['spread_code', 'spread_type'])
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Generated {len(df)} spread combinations, saved to {output_file}")
    return df

# Usage example
if __name__ == "__main__":
    # Get all contracts from merged data
    try:
        merged_data = pd.read_feather("./merged.feather")
        existing_contracts = [col for col in merged_data.columns if col != 'timestamp']
        print(f"Read {len(existing_contracts)} contracts from merged data")
    except Exception as e:
        print(f"Failed to read merged data: {str(e)}")
        print("Using example contract list...")
        # Example contract list
        existing_contracts = [
            'I2409', 'I2410', 'I2411', 'I2412',
            'I2501', 'I2502', 'I2503', 'I2504', 'I2505', 'I2506', 'I2507', 'I2508', 'I2509'
        ]
    
    # Main month configuration
    main_months = [1, 5, 9]
    
    # Generate and save spread list
    output_file = "./iron_spreads.csv"
    spread_df = generate_all_spreads(existing_contracts, main_months, output_file)
    
    # Display sample results
    print("\nSpread list sample:")
    print(spread_df[['spread_type', 'spread_code', 'contract_a', 'contract_b']].head(10).to_string(index=False)) 