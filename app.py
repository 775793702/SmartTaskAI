"""
SmartTask AI - 智能任务管理系统
基于Flask + 通义千问大模型
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import dashscope
from dashscope import Generation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# ==================== 初始化配置 ====================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 加载环境变量
load_dotenv()

# 设置阿里云通义千问API密钥
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if API_KEY:
    dashscope.api_key = API_KEY
    print(f"✓ API密钥已加载，长度: {len(API_KEY)}")
else:
    print("⚠️ 警告: 未找到API密钥，将使用备用分析模式")
    print("请在项目根目录创建 .env 文件，内容为: DASHSCOPE_API_KEY=您的密钥")

# ==================== 数据库模型 ====================
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    estimated_time = db.Column(db.String(50))
    sub_tasks = db.Column(db.String(500))
    difficulty = db.Column(db.String(20))
    energy_level = db.Column(db.String(20))
    focus_required = db.Column(db.String(10))
    priority = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_completed = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'description': self.description,
            'category': self.category,
            'estimated_time': self.estimated_time,
            'sub_tasks': json.loads(self.sub_tasks) if self.sub_tasks else [],
            'difficulty': self.difficulty,
            'energy_level': self.energy_level,
            'focus_required': self.focus_required,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'is_completed': self.is_completed
        }

# ==================== AI分析模块 ====================
# 训练一个简单的本地分类器作为备用
sample_descriptions = [
    "完成项目报告并提交给经理", "学习机器学习第三章", "去超市买牛奶和鸡蛋",
    "准备下周的会议PPT", "阅读《百年孤独》第50页", "预约周六的牙医",
    "调试程序中的bug", "缴纳水电费", "复习线性代数", "给家人打电话",
    "编写Python爬虫程序", "健身一小时", "整理房间大扫除", "学习英语单词",
    "完成客户需求分析", "购买生日礼物", "准备考试复习资料", "修复网站bug"
]
sample_categories = ['工作', '学习', '生活', '工作', '学习', '生活', 
                     '工作', '生活', '学习', '生活', '工作', '生活',
                     '生活', '学习', '工作', '生活', '学习', '工作']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(sample_descriptions)
classifier = MultinomialNB()
classifier.fit(X_train, sample_categories)

def get_fallback_analysis(description):
    """
    智能备用分析方案 - 当大模型API不可用时使用
    基于关键词规则和简单逻辑推理
    """
    print(f"[备用方案] 开始智能分析: {description}")
    
    # 预处理：转为小写，方便匹配
    desc_lower = description.lower()
    words = desc_lower.split()
    word_count = len(words)
    
    # ============ 1. 智能分类 ============
    # 定义关键词映射
    work_keywords = ["工作", "项目", "会议", "汇报", "ppt", "报告", "客户", "经理", "老板", 
                     "演示", "商务", "谈判", "合同", "邮件", "deadline", "截止", "提交", 
                     "方案", "设计", "开发", "编程", "代码", "bug", "测试", "部署"]
    
    study_keywords = ["学习", "复习", "考试", "作业", "论文", "课程", "阅读", "预习", 
                      "练习", "训练", "实验", "研究", "学术", "知识", "技能", "掌握", 
                      "理解", "记忆", "背诵", "预习", "复习", "刷题", "模拟"]
    
    life_keywords = ["购物", "买菜", "做饭", "打扫", "清洁", "洗衣", "整理", "休息", 
                     "睡觉", "电影", "音乐", "游戏", "运动", "健身", "跑步", "瑜伽", 
                     "旅游", "旅行", "约会", "聚会", "聚餐", "家庭", "家人", "朋友", 
                     "娱乐", "放松", "休闲", "看病", "医院", "医生", "健康"]
    
    # 计算关键词匹配分数
    work_score = sum(1 for keyword in work_keywords if keyword in desc_lower)
    study_score = sum(1 for keyword in study_keywords if keyword in desc_lower)
    life_score = sum(1 for keyword in life_keywords if keyword in desc_lower)
    
    # 根据分数和关键词确定分类
    if "机器学习" in description or "人工智能" in description or "ai" in desc_lower:
        category = "学习"  # AI相关优先归为学习
    elif "ppt" in desc_lower or "演示" in description or "汇报" in description:
        category = "工作"
    elif work_score >= study_score and work_score >= life_score:
        category = "工作"
    elif study_score >= work_score and study_score >= life_score:
        category = "学习"
    elif life_score >= work_score and life_score >= study_score:
        category = "生活"
    else:
        # 如果分数都很低，使用基于描述的启发式判断
        if any(word in ["我", "我们", "自己", "个人"] for word in words[:3]):
            category = "生活" if len(words) < 8 else "学习"
        else:
            category = "其他"
    
    # ============ 2. 智能耗时预估 ============
    # 基于多个因素：字数、关键词复杂度、任务类型
    complexity_indicators = 0
    
    # 检测复杂任务关键词
    complex_indicators = ["项目", "报告", "论文", "系统", "应用", "程序", "开发", 
                          "设计", "分析", "研究", "机器学习", "深度学习", "人工智能"]
    for indicator in complex_indicators:
        if indicator in description:
            complexity_indicators += 1
    
    # 检测紧急/截止时间
    urgent_indicators = ["明天", "今天", "紧急", "立刻", "马上", "尽快", "截止", "deadline"]
    is_urgent = any(indicator in description for indicator in urgent_indicators)
    
    # 估算时间
    if complexity_indicators >= 2:
        estimated_time = "半天-1天"
    elif complexity_indicators == 1:
        estimated_time = "2-4小时"
    elif word_count > 15:
        estimated_time = "1-2小时"
    elif word_count > 8:
        estimated_time = "1小时内"
    elif word_count > 4:
        estimated_time = "30分钟内"
    else:
        estimated_time = "15分钟内"
    
    # 如果是紧急任务，适当缩短预估时间
    if is_urgent and "天" in estimated_time:
        estimated_time = "半天"
    elif is_urgent and "小时" in estimated_time:
        estimated_time = estimated_time.split("-")[0]  # 取较短时间
    
    # ============ 3. 智能生成子任务 ============
    # 基于任务类型和描述生成相关子任务
    base_templates = {
        "工作": ["收集相关材料", "制定执行计划", "完成核心部分", "检查与完善"],
        "学习": ["阅读相关资料", "整理笔记重点", "完成练习/实践", "复习总结"],
        "生活": ["准备所需物品", "规划时间路线", "执行主要活动", "整理收尾"],
        "其他": ["明确目标需求", "分步骤执行", "检查完成情况", "记录总结"]
    }
    
    # 从描述中提取名词作为具体化元素
    try:
        blob = TextBlob(description)
        nouns = [str(noun) for noun in blob.noun_phrases if len(str(noun)) > 1]
    except:
        nouns = []
    
    # 选择基础模板
    template = base_templates.get(category, base_templates["其他"])
    
    # 如果有提取到名词，尝试具体化子任务
    if nouns and len(nouns) >= 2:
        # 使用前两个名词来具体化
        noun1, noun2 = nouns[0], nouns[1] if len(nouns) > 1 else nouns[0]
        
        if category == "工作":
            sub_tasks = [
                f"收集关于{noun1}的资料",
                f"制定{noun1}相关的计划",
                f"完成{noun1}的核心部分"
            ]
        elif category == "学习":
            sub_tasks = [
                f"学习{noun1}的基础知识",
                f"练习{noun1}的相关应用",
                f"总结{noun1}的重点内容"
            ]
        elif category == "生活":
            sub_tasks = [
                f"准备{noun1}相关物品",
                f"安排{noun1}的时间",
                f"完成{noun1}的活动"
            ]
        else:
            sub_tasks = template[:3]
    else:
        # 使用通用模板
        sub_tasks = template[:3]
    
    # 确保子任务数量在2-3个
    sub_tasks = sub_tasks[:3]
    if len(sub_tasks) < 2:
        sub_tasks.extend(["补充执行步骤", "最终检查完善"])
    
    # ============ 4. 评估任务难度 ============
    if complexity_indicators >= 2 or word_count > 20:
        difficulty = "困难"
    elif complexity_indicators == 1 or word_count > 10:
        difficulty = "中等"
    else:
        difficulty = "简单"
    
    # ============ 5. 评估精力消耗 ============
    if category == "工作" and complexity_indicators >= 1:
        energy_level = "高"
    elif category == "学习" and "复习" in description or "考试" in description:
        energy_level = "高"
    elif category == "生活" and ("运动" in description or "健身" in description):
        energy_level = "高"
    elif word_count > 12:
        energy_level = "中"
    else:
        energy_level = "低"
    
    # ============ 6. 判断是否需要专注 ============
    focus_keywords = ["专注", "集中", "认真", "仔细", "深入", "分析", "思考", 
                      "计算", "编程", "写作", "设计", "创造"]
    mental_tasks = ["学习", "工作", "研究", "分析", "编程", "写作", "设计"]
    
    if any(keyword in description for keyword in focus_keywords):
        focus_required = "是"
    elif category in ["工作", "学习"] and word_count > 8:
        focus_required = "是"
    elif "放松" in description or "休息" in description or "娱乐" in description:
        focus_required = "否"
    else:
        focus_required = "是" if difficulty in ["中等", "困难"] else "否"
    
    # ============ 7. 评估优先级 ============
    urgent_words = ["紧急", "立刻", "马上", "尽快", "立即", "今天", "明天", "尽快完成"]
    important_words = ["重要", "关键", "必须", "必要", "优先", "主要", "首要"]
    
    if any(word in description for word in urgent_words):
        priority = "紧急"
    elif any(word in description for word in important_words):
        priority = "高"
    elif category == "工作" or "截止" in description:
        priority = "高"
    elif category == "学习" and ("考试" in description or "作业" in description):
        priority = "高"
    elif category == "生活" and ("健康" in description or "医疗" in description):
        priority = "高"
    else:
        priority = "中"
    
    # 构建分析结果
    result = {
        "category": category,
        "estimated_time": estimated_time,
        "sub_tasks": sub_tasks,
        "difficulty": difficulty,
        "energy_level": energy_level,
        "focus_required": focus_required,
        "priority": priority
    }
    
    print(f"[备用方案] 分析完成: {category}类, {estimated_time}, 难度:{difficulty}")
    return result

def ai_analyze_task(description):
    """使用通义千问大模型进行智能任务分析"""
    print(f"\n{'='*50}")
    print(f"[AI分析] 开始分析任务: {description}")
    
    # 检查API密钥
    if not dashscope.api_key:
        print("[AI分析] 错误: API密钥未设置，使用备用方案")
        return get_fallback_analysis(description)
    
    # 检查描述长度
    if len(description.strip()) < 3:
        print("[AI分析] 任务描述过短，使用备用方案")
        return get_fallback_analysis(description)
    
    # 精心设计的Prompt
    prompt = f"""你是一个专业的任务规划师。请严格按以下规则分析任务：

【分类规则】
- 工作：与职业、项目、会议、汇报、商务、客户相关
- 学习：与知识获取、技能提升、课程作业、考试、阅读、培训相关  
- 生活：与日常起居、购物、健康、家庭、娱乐、运动、社交相关
- 其他：不符合以上分类

【任务描述】{description}

请返回一个严格的JSON对象，包含以下字段：
1. category: 从["工作","学习","生活","其他"]中选择
2. estimated_time: 从["15分钟内","30分钟内","1小时内","1-2小时","2-4小时","半天","全天","1-2天","一周以上"]中选择
3. sub_tasks: 一个包含2-3个具体、可立即执行步骤的数组
4. difficulty: 从["简单","中等","困难"]中选择
5. energy_level: 从["低","中","高"]中选择
6. focus_required: 从["是","否"]中选择
7. priority: 从["低","中","高","紧急"]中选择

示例1：任务"完成机器学习大作业"
{{"category":"学习","estimated_time":"2-4小时","sub_tasks":["收集相关数据集","实现基础模型","撰写实验报告"],"difficulty":"困难","energy_level":"高","focus_required":"是","priority":"高"}}

示例2：任务"去超市买周末食材"
{{"category":"生活","estimated_time":"1小时内","sub_tasks":["列出购物清单","前往超市","分类整理食材"],"difficulty":"简单","energy_level":"低","focus_required":"否","priority":"中"}}

示例3：任务"准备下周团队会议报告"
{{"category":"工作","estimated_time":"2-4小时","sub_tasks":["收集上周数据","制作PPT幻灯片","预演汇报内容"],"difficulty":"中等","energy_level":"中","focus_required":"是","priority":"高"}}

请只返回JSON对象，不要有任何其他解释。"""
    
    try:
        print("[AI分析] 调用通义千问API...")
        response = Generation.call(
            model="qwen-max",  # 使用最强大的模型
            prompt=prompt,
            seed=1234,  # 固定随机种子，使结果更稳定
            temperature=0.1,  # 低温度，减少随机性
            result_format='message'
        )
        
        print(f"[AI分析] API响应状态码: {response.status_code}")
        
        if response.status_code == 200 and response.output:
            result_text = response.output.choices[0].message.content
            print(f"[AI分析] 原始返回文本: {result_text[:200]}...")
            
            # 清理响应文本，提取JSON部分
            json_pattern = r'```json\s*(.*?)\s*```|({.*})'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                # 提取JSON字符串
                json_str = match.group(1) if match.group(1) else match.group(2)
                json_str = json_str.strip()
                
                print(f"[AI分析] 提取的JSON: {json_str}")
                
                # 解析JSON
                analysis = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ["category", "estimated_time", "sub_tasks", 
                                 "difficulty", "energy_level", "focus_required", "priority"]
                
                if all(field in analysis for field in required_fields):
                    print(f"[AI分析] ✓ 大模型分析成功!")
                    print(f"      类别: {analysis['category']}")
                    print(f"      预估时间: {analysis['estimated_time']}")
                    print(f"      难度: {analysis['difficulty']}")
                    return analysis
                else:
                    print(f"[AI分析] ✗ API返回字段不全，使用备用方案")
                    print(f"      缺失字段: {[f for f in required_fields if f not in analysis]}")
            else:
                print(f"[AI分析] ✗ 未找到JSON，使用备用方案")
                
        else:
            print(f"[AI分析] ✗ API响应异常: {response}")
                
    except Exception as e:
        print(f"[AI分析] ✗ 调用大模型API失败: {type(e).__name__}: {e}")
    
    # 如果API调用失败，使用备用方案
    print("[AI分析] 切换到备用分析方案")
    return get_fallback_analysis(description)

# ==================== 路由定义 ====================
@app.route('/')
def index():
    """提供前端页面"""
    return render_template('index.html')

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务"""
    tasks = Task.query.order_by(Task.created_at.desc()).all()
    return jsonify([task.to_dict() for task in tasks])

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """创建新任务，并调用AI分析"""
    data = request.json
    description = data.get('description', '').strip()
    
    if not description:
        return jsonify({'error': '任务描述不能为空'}), 400
    
    print(f"\n{'='*50}")
    print(f"[创建任务] 接收到新任务: {description}")
    
    # 调用AI进行分析
    ai_result = ai_analyze_task(description)
    
    print(f"[创建任务] AI分析结果:")
    print(f"  类别: {ai_result['category']}")
    print(f"  时间: {ai_result['estimated_time']}")
    print(f"  子任务: {ai_result['sub_tasks']}")
    
    # 创建任务对象
    new_task = Task(
        description=description,
        category=ai_result['category'],
        estimated_time=ai_result['estimated_time'],
        sub_tasks=json.dumps(ai_result['sub_tasks'], ensure_ascii=False),
        difficulty=ai_result.get('difficulty', '中等'),
        energy_level=ai_result.get('energy_level', '中'),
        focus_required=ai_result.get('focus_required', '是'),
        priority=ai_result.get('priority', '中')
    )
    
    try:
        db.session.add(new_task)
        db.session.commit()
        print(f"[创建任务] ✓ 任务保存成功，ID: {new_task.id}")
        
        return jsonify({
            'id': new_task.id,
            'message': '任务创建成功',
            'ai_analysis': ai_result
        })
    except Exception as e:
        db.session.rollback()
        print(f"[创建任务] ✗ 数据库保存失败: {e}")
        return jsonify({'error': '保存任务失败'}), 500

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """删除任务"""
    task = Task.query.get(task_id)
    if task:
        db.session.delete(task)
        db.session.commit()
        print(f"[删除任务] ✓ 任务 {task_id} 已删除")
        return jsonify({'message': '任务已删除'})
    return jsonify({'error': '任务未找到'}), 404

@app.route('/api/tasks/<int:task_id>/toggle', methods=['PUT'])
def toggle_task(task_id):
    """切换任务完成状态"""
    task = Task.query.get(task_id)
    if task:
        task.is_completed = not task.is_completed
        db.session.commit()
        status = "已完成" if task.is_completed else "未完成"
        print(f"[切换状态] ✓ 任务 {task_id} 状态改为: {status}")
        return jsonify({'message': '状态已更新', 'is_completed': task.is_completed})
    return jsonify({'error': '任务未找到'}), 404

@app.route('/api/test', methods=['GET'])
def test_api():
    """测试接口"""
    return jsonify({
        'status': 'online',
        'service': 'SmartTask AI',
        'ai_enabled': bool(dashscope.api_key),
        'timestamp': datetime.utcnow().isoformat()
    })

# ==================== 应用初始化 ====================
def init_database():
    """初始化数据库"""
    with app.app_context():
        db.create_all()
        print("✓ 数据库初始化完成")

def test_ai_connection():
    """测试AI连接"""
    if dashscope.api_key:
        print("✓ AI功能已启用 (通义千问)")
    else:
        print("⚠️ AI功能使用备用模式 (本地规则)")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SmartTask AI - 智能任务管理系统")
    print("="*60)
    
    # 初始化
    init_database()
    test_ai_connection()
    
    print(f"\n服务启动信息:")
    print(f"  - 本地访问: http://127.0.0.1:5000")
    print(f"  - API测试: http://127.0.0.1:5000/api/test")
    print(f"  - 按 Ctrl+C 停止服务")
    print("="*60 + "\n")
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)