import json
import os

def create_initial_config():
    # 현재 최적화된 기본 설정값
    default_params = {
        'outer_margin': 10,    
        'inner_margin': 10,    
        'min_area': 1050,       
        'max_area': 5000,      
        'min_short_side': 40,  
        'max_short_side': 100,  
        'min_long_side': 40,   
        'max_long_side': 100,   
        'min_ratio': 0.5,      
        'max_ratio': 1.5, 
        'min_dist': 150,
        'max_dist': 210,
        'min_edge_length_sum': 240, 
        'dilation_iter': 3     
    }

    config_data = {}
    
    # 1번부터 25번까지 기본값으로 초기화
    for i in range(1, 26):
        config_data[str(i)] = default_params.copy()

    # [수정] 현재 스크립트 위치 기준으로 절대 경로 생성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "product_config.json")
    
    # 폴더가 없으면 생성 (이제 base_dir이 유효한 경로이므로 에러 안 남)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
        
    print(f"Configuration file created at: {save_path}")

if __name__ == "__main__":
    create_initial_config()