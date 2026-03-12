use pyo3::prelude::*;
use std::collections::HashMap;

const EOS: char = '\0';

/// 默认的排序规则
fn default_orders() -> HashMap<String, Vec<String>> {
    let mut orders = HashMap::new();
    orders.insert("date".to_string(), vec!["year".to_string(), "month".to_string(), "day".to_string()]);
    orders.insert("fraction".to_string(), vec!["denominator".to_string(), "numerator".to_string()]);
    orders.insert("measure".to_string(), vec!["denominator".to_string(), "numerator".to_string(), "value".to_string()]);
    orders.insert("money".to_string(), vec!["value".to_string(), "currency".to_string()]);
    orders.insert("time".to_string(), vec!["noon".to_string(), "hour".to_string(), "minute".to_string(), "second".to_string()]);
    orders
}

/// Token 结构体，表示解析后的标记
#[derive(Debug, Clone)]
struct Token {
    name: String,
    order: Vec<String>,
    members: HashMap<String, String>,
}

impl Token {
    fn new(name: String) -> Self {
        Token {
            name,
            order: Vec::new(),
            members: HashMap::new(),
        }
    }

    fn append(&mut self, key: String, value: String) {
        self.order.push(key.clone());
        self.members.insert(key, value);
    }

    fn to_string(&self, orders: &HashMap<String, Vec<String>>) -> String {
        let mut output = format!("{} {{", self.name);
        
        let mut current_order = self.order.clone();
        
        // 如果存在预定义的顺序且不需要保留原顺序
        if let Some(predefined_order) = orders.get(&self.name) {
            if self.members.get("preserve_order") != Some(&"true".to_string()) {
                current_order = predefined_order.clone();
            }
        }

        for key in &current_order {
            if let Some(value) = self.members.get(key) {
                output.push_str(&format!(r#" {}: "{}""#, key, value));
            }
        }
        
        output.push_str(" }");
        output
    }
}

/// Reorder 类 - 用于解析和重新排序 token
#[pyclass]
#[derive(Debug)]
pub struct Reorder {
    orders: HashMap<String, Vec<String>>,
    index: usize,
    text: String,
    current_char: char,
    tokens: Vec<Token>,
}

impl Reorder {
    /// 加载输入文本并初始化解析器状态
    fn load(&mut self, input: &str) -> PyResult<()> {
        if input.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Input cannot be empty"));
        }
        
        self.index = 0;
        self.text = input.to_string();
        self.current_char = self.text.chars().next().unwrap_or(EOS);
        self.tokens.clear();
        
        Ok(())
    }

    /// 读取下一个字符
    fn read(&mut self) -> bool {
        if self.index < self.text.len().saturating_sub(1) {
            self.index += 1;
            self.current_char = self.text.chars().nth(self.index).unwrap_or(EOS);
            true
        } else {
            self.current_char = EOS;
            false
        }
    }

    /// 解析空白字符
    fn parse_ws(&mut self) -> bool {
        let mut not_eos = self.current_char != EOS;
        while not_eos && self.current_char == ' ' {
            not_eos = self.read();
        }
        not_eos
    }

    /// 解析期望的字符
    fn parse_char(&mut self, expected: char) -> bool {
        if self.current_char == expected {
            self.read();
            true
        } else {
            false
        }
    }

    /// 解析一组字符中的任意一个
    fn parse_chars(&mut self, expected: &[char]) -> bool {
        let mut ok = false;
        for ch in expected.iter() {
            ok |= self.parse_char(*ch);
        }
        ok
    }

    /// 解析 key（由 ASCII 字母和下划线组成）
    fn parse_key(&mut self) -> PyResult<String> {
        if self.current_char == EOS {
            return Err(pyo3::exceptions::PyAssertionError::new_err("Unexpected EOS"));
        }
        if self.current_char.is_whitespace() {
            return Err(pyo3::exceptions::PyAssertionError::new_err("Unexpected whitespace"));
        }

        let mut key = String::new();
        while self.current_char.is_ascii_alphabetic() || self.current_char == '_' {
            key.push(self.current_char);
            self.read();
        }
        Ok(key)
    }

    /// 解析值（双引号包围的字符串，支持转义）
    fn parse_value(&mut self) -> PyResult<String> {
        if self.current_char == EOS {
            return Err(pyo3::exceptions::PyAssertionError::new_err("Unexpected EOS"));
        }

        let mut value = String::new();

        while self.current_char != '"' {
            if self.current_char == EOS {
                return Err(pyo3::exceptions::PyValueError::new_err("Unterminated string"));
            }
            
            let escape = self.current_char == '\\';
            value.push(self.current_char);
            self.read();
            
            if escape {
                if self.current_char == EOS {
                    return Err(pyo3::exceptions::PyValueError::new_err("Unexpected escape at EOS"));
                }
                value.push(self.current_char);
                self.read();
            }
        }
        
        Ok(value)
    }

    /// 解析输入文本为 tokens
    fn parse(&mut self, input: &str) -> PyResult<()> {
        self.load(input)?;
        
        while self.parse_ws() {
            let name = self.parse_key()?;
            self.parse_chars(&[' ', '{', ' ']);

            let mut token = Token::new(name);
            
            while self.parse_ws() {
                if self.current_char == '}' {
                    self.parse_char('}');
                    break;
                }
                
                let key = self.parse_key()?;
                self.parse_chars(&[':']);
                self.parse_ws();
                self.parse_char('"');
                let value = self.parse_value()?;
                self.parse_char('"');
                token.append(key, value);
            }
            
            self.tokens.push(token);
        }
        
        Ok(())
    }
}

#[pymethods]
impl Reorder {
    #[new]
    fn new() -> Self {
        Reorder {
            orders: default_orders(),
            index: 0,
            text: String::new(),
            current_char: EOS,
            tokens: Vec::new(),
        }
    }

    /// 重新排序输入文本中的 tokens
    ///
    /// Args:
    ///     input: 输入文本，格式为 "name { key: \"value\" ... } ..."
    ///
    /// Returns:
    ///     重新排序后的文本
    ///
    /// Example:
    ///     >>> reorder = Reorder()
    ///     >>> reorder.reorder(r#"money { currency: \"USD\" value: \"100\" }"#)
    ///     'money { value: "100" currency: "USD" }'
    fn reorder(&mut self, input: &str) -> PyResult<String> {
        self.parse(input)?;
        
        let mut output = String::new();
        for token in &self.tokens {
            if !output.is_empty() {
                output.push(' ');
            }
            output.push_str(&token.to_string(&self.orders));
        }
        
        Ok(output)
    }

    /// 获取当前的排序规则
    #[getter]
    fn get_orders(&self) -> HashMap<String, Vec<String>> {
        self.orders.clone()
    }

    /// 设置排序规则
    #[setter]
    fn set_orders(&mut self, orders: HashMap<String, Vec<String>>) {
        self.orders = orders;
    }
}

pub fn register_module(core_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let reorder_module = PyModule::new(core_module.py(), "reorder")?;
    reorder_module.add_class::<Reorder>()?;
    core_module.add_submodule(&reorder_module)?;
    Ok(())
}
