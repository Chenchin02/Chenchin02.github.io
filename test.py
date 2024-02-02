from dotenv import dotenv_values

# 临时打印文件内容来调试
env_values = dotenv_values('venv/.env')
print(env_values)  # 看看实际加载了什么

api_key = env_values.get('\ufeffAPI_KEY') or env_values.get('API_KEY')
print(api_key)