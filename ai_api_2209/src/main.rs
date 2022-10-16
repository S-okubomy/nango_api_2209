use rusqlite::{params, Connection, Result};
use chrono::{DateTime, Utc, Local};
use serde_json::{json, Value};


const STR_PKEY: &str = "nango7_ai_nango_kun";

#[derive(Debug)]
struct StudyQa1 {
    id: u16,
    is_positive: bool,
    exp_val: String,
    obj_val: String,
    ins_date: DateTime<Local>,
    update_date: DateTime<Local>,
    del_flag: bool,
}

fn open_my_db() -> Result<Connection,rusqlite::Error> {
    let path = "./db/ai_db.db";
    let con = Connection::open(&path)?;
    println!("{}", con.is_autocommit());
    Ok(con)
}

fn select_all(con:&Connection){
    let mut stmt = con.prepare("select id,is_positive,exp_val,obj_val,ins_date,update_date,del_flag from study_qa1").unwrap();
    let qas = stmt.query_map(params![], |row| {
      Ok(StudyQa1 {
          id: row.get(0).unwrap(),
          is_positive: row.get(1).unwrap(),
          exp_val: row.get(2).unwrap(),
          obj_val: row.get(3).unwrap(),
          ins_date: row.get(4).unwrap(),
          update_date: row.get(5).unwrap(),
          del_flag: row.get(6).unwrap(),
      })
    }).unwrap();

    for q in qas {
      println!("{:?}", q.unwrap());
    }

}

fn main() {
    let con = open_my_db().unwrap();
    select_all(&con);

    // TODO csvの学習データ読み込み

    // TODO 学習データをsqliteのDBに登録

    // TODO 機械学習処理作成
}


#[derive(Debug)]
enum ExecMode {
    Learn,
    Predict { que_sentence: String },
}

impl ExecMode {
    fn new(event: Value) -> Result<ExecMode, String> {
        let mode: &str = event["mode"].as_str().unwrap_or("");
        let que_sentence = event["que_sentence"].as_str().unwrap_or("");
        let pkey = event["pkey"].as_str().unwrap_or("");

        if pkey.len() == 0 || pkey != STR_PKEY {
            return Err("Not executable".to_string());
        }

        match mode {
            "l" => {
                Ok(ExecMode::Learn)
            },
            "p" => {
                if que_sentence.len() > 0 {
                    Ok(ExecMode::Predict { que_sentence: que_sentence.to_string() })
                } else {
                    Err("予測時は、質問文を入力してください。".to_string())
                }
            },
            _ => {
                Err("学習: l、予測: p を指定してください。".to_string())
            }
        }
    }

    fn run(self) -> Value {
        match self {
            ExecMode::Learn => {
                // learn()
            },
            ExecMode::Predict { que_sentence } => {
                // predict(que_sentence)
            },
        }

    }
}


// fn learn() -> Value {
//     let qa_data: QaData = read_csv().unwrap_or_else(|err| {
//         println!("error running read: {}", err);
//         std::process::exit(1);
//     });

//     let mut docs: Vec<Vec<String>> = Vec::new();
//     for input_qa in qa_data.que_vec {
//         let doc_vec: Vec<String> = get_tokenizer(input_qa);
//         docs.push(doc_vec);
//     }

//     out_csv_word(&docs).unwrap_or_else(|err| {
//         println!("error running out_csv_word csv: {}", err);
//         std::process::exit(1);
//     });

//     let tf_idf_res = tf_idf::TfIdf::get_tf_idf(&docs);
//     // 学習済みモデル出力
//     out_csv(tf_idf_res).unwrap_or_else(|err| {
//         println!("error running output csv: {}", err);
//         std::process::exit(1);
//     });

//     let res_json: Value = json!({
//         "code": 200,
//         "success": true,
//         "mode": "learn",
//     });
//     res_json
// }

// fn predict(que_sentence: String) -> Value {
//     let qa_data: QaData = read_csv().unwrap_or_else(|err| {
//         println!("error running read: {}", err);
//         std::process::exit(1);
//     });

//     let docs: Vec<Vec<String>> = read_word_list_csv().unwrap_or_else(|err| {
//         println!("error running read: {}", err);
//         std::process::exit(1);
//     });

//     let tfidf: tf_idf::TfIdf = read_model_csv().unwrap();
//     let trg: Vec<String> = get_tokenizer(que_sentence.to_owned());
//     let ans_vec: Vec<(usize, f64)> = tf_idf::TfIdf::predict(tfidf, &docs, &trg);

//     let res_json: Value = make_json(que_sentence, qa_data, ans_vec);
//     res_json
// }
