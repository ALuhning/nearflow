import { CustomLink } from "@/customization/components/custom-link";
import * as Form from "@radix-ui/react-form";
import "ace-builds/src-noconflict/ext-language_tools";
import "ace-builds/src-noconflict/mode-python";
import "ace-builds/src-noconflict/theme-github";
import "ace-builds/src-noconflict/theme-twilight";
import BaseModal from "../baseModal";
import { Textarea } from "@/components/ui/textarea";
import { ADD_TO_MARKETPLACE_SUBTITLE } from "@/constants/constants";
import { cn } from "@/utils/utils";
import useFlowStore from "@/stores/flowStore";
import { ReactNode, useState } from "react";
import { useShallow } from "zustand/react/shallow";

export default function ApiModal({
    children,
    open: myOpen,
    setOpen: mySetOpen,
}: {
    children: ReactNode;
    open?: boolean;
    setOpen?: (a: boolean | ((o?: boolean) => boolean)) => void;
}) {
    const maxLength = 50
    const descriptionMaxLength = 250
    const minLength = 1

    const [open, setOpen] =
        mySetOpen !== undefined && myOpen !== undefined
            ? [myOpen, mySetOpen]
            : useState(false);
    const [name, setName] = useState("")
    const [description, setDescription] = useState("")
    const [isMaxLength, setIsMaxLength] = useState(false);
    const [isMaxDescriptionLength, setIsMaxDescriptionLength] = useState(false);
    const [isMinLength, setIsMinLength] = useState(false);

    const currentFlowId = useFlowStore(
        useShallow((state) => state.currentFlow?.id),
    );

    const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { value } = event.target;
        if (value.length >= maxLength) {
            setIsMaxLength(true);
        } else {
            setIsMaxLength(false);
        }
        if (value.length < minLength) {
            setIsMinLength(true);
        } else {
            setIsMinLength(false);
        }
        setName!(value);
        if (value.length === 0) {
            setIsMinLength(true);
        }
    };

    const handleDescriptionChange = (
        event: React.ChangeEvent<HTMLTextAreaElement>,
    ) => {
        const { value } = event.target;
        if (value.length >= descriptionMaxLength) {
            setIsMaxDescriptionLength(true);
        } else {
            setIsMaxDescriptionLength(false);
        }
        setDescription!(value);
    };


    //submits the form if someone hits enter , not sure we want this behaviour
    const handleDescriptionKeyDown = (
        event: React.KeyboardEvent<HTMLTextAreaElement>,
    ) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            console.log("submit")
        }
        // else allow default (newline)
    };

    const handleFocus = (event) => event.target.select();

    return (
        <BaseModal
            closeButtonClassName="!top-3"
            open={open}
            setOpen={setOpen}
            size="medium"
            className="pt-4"
        >
            <BaseModal.Header description={ADD_TO_MARKETPLACE_SUBTITLE}>
                <span className="pr-2">Add to Marketplace</span>
            </BaseModal.Header>
            <BaseModal.Content>
                <Form.Root>
                    <Form.Field name="description">
                        <Form.Label>
                            Flow Id:{" "}
                            <span style={{ color: "grey" }}>{currentFlowId}
                            </span>
                        </Form.Label>
                        <div style={{ marginTop: "20px" }}>
                            <Form.Label>
                                Description:
                            </Form.Label>
                            <Form.Control asChild>
                                <Textarea
                                    name="flowId"
                                    value={description}
                                    placeholder={"Add a description here to advertise your agent"}
                                    onKeyDown={handleDescriptionKeyDown}
                                    maxLength={2000}
                                    onDoubleClickCapture={handleFocus}
                                    id="flowId"
                                    onChange={handleDescriptionChange}
                                    rows={4}
                                />
                            </Form.Control>
                        </div>
                    </Form.Field>
                </Form.Root>
            </BaseModal.Content>

            <BaseModal.Footer
                submit={{
                    label: "Add",
                }}
            />
        </BaseModal >
    );
}
