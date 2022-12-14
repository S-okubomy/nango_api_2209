CREATE TABLE IF NOT EXISTS study_qa1
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  is_positive boolean not null DEFAULT TRUE,
  exp_val text,
  obj_val text,
  ins_date timestamp with time zone not null,
  update_date timestamp with time zone not null,
  del_flag boolean not null DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS output_word_list
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  words text,
  ins_date timestamp with time zone not null,
  update_date timestamp with time zone not null,
  del_flag boolean not null DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS output_model
(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  vals text,
  ins_date timestamp with time zone not null,
  update_date timestamp with time zone not null,
  del_flag boolean not null DEFAULT FALSE
);