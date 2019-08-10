import React from 'react';

/*******************************************************************************
  <Button /> Component
*******************************************************************************/

class Button extends React.Component {
  render() {
    const { enabled, onClick } = this.props;

    return (
    <button type="button" disabled={!enabled} className="btn btn--icon-disclosure" onClick={onClick}>Run
      <svg>
        <use xlinkHref="#icon__disclosure"></use>
      </svg>
    </button>
    );
  }
}

export default Button;
