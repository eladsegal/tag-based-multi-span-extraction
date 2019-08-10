import React from 'react';
import { mapToArray } from '../../../utils'
import {
    Button
  } from 'reactstrap';

class DatasetLocalSelector extends React.Component {
    constructor(props) {
        super(props);
        this.fileInputRef = React.createRef();
        this.simulateClick = this.simulateClick.bind(this);
        this.change = this.change.bind(this);
        this.state = {
            dataset: undefined,
            file: undefined
        };
    }

    shouldComponentUpdate(nextProps, nextState) {
        return this.state.dataset !== nextState.dataset;
    }

    componentDidUpdate(prevProps, prevState) {
        if (!prevState || prevState.dataset !== this.state.dataset) {
            this.props.onChange(this.state.dataset);
        }
    }
    
    simulateClick() {
        this.fileInputRef.current.click();
    }

    change(files) {
        if (files.length > 0) {
            const file = files[0]
            if (file) {
                const reader = new FileReader();
                reader.onloadend = (e) => {
                    const array = mapToArray(JSON.parse(e.target.result), 'passage_id', 'passage_index');
                    this.setState({ 
                        file: file,
                        dataset: array 
                    });
                };
                reader.readAsText(file);
              }
        } else {
            this.setState({ dataset: undefined })
        }
    }

    render() {      
        return <div>
            <input ref={this.fileInputRef} style={{'display': 'none'}} type='file' id='file' accept='.json' onChange={ (e) => this.change(e.target.files) } />
            <Button size='md' onClick={this.simulateClick}>
                Choose File
            </Button>
            {this.state.file ? this.state.file.name : ''}
        </div>
    }
}

export default DatasetLocalSelector;
