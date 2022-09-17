use rusqlite::{params, Connection, Result};
use chrono::{DateTime, Utc, Local};

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
