import React from 'react';

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;

    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    var answer = outputs['answer'];

    return (
      <div className="model__content">

       {/*
         * TODO: This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}

        <div className="form__field">
          <label>Answer</label>
          <div className="model__content__summary"><pre>{ JSON.stringify(answer, null, 2) }</pre></div>
        </div>

      </div>
    );
  }
}

export default ModelOutput;
