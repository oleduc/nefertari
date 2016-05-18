for(int i = 0; i < ctx._source[field_name].size(); i++){
  if((int)ctx._source[field_name][i].id == (int)nested_document.id){
    ctx._source[field_name][i] = nested_document; return true;
  }
}